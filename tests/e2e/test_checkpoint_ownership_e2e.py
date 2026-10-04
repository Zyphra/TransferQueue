# Copyright 2025 The TransferQueue Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Real-service restore controls with a deterministic controller/storage cut."""

import asyncio
import json
import pickle
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest
import ray
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

import transfer_queue as tq
from transfer_queue import interface


@pytest.fixture(scope="module")
def owned_ray():
    assert not ray.is_initialized()
    ray.init(address="local", num_cpus=4, num_gpus=0, object_store_memory=256 * 1024 * 1024)
    yield
    ray.shutdown()


@pytest.fixture
def services(owned_ray):
    config = OmegaConf.create(
        {
            "controller": {"polling_mode": True},
            "backend": {
                "storage_backend": "SimpleStorage",
                "SimpleStorage": {"total_storage_size": 200, "num_data_storage_units": 2},
            },
        }
    )
    tq.init(config)
    yield config
    tq.close()


def put(key, field="value", partition="train", value=1):
    tq.kv_put(key=key, partition_id=partition, fields=TensorDict({field: torch.tensor([[value]])}, batch_size=1))


def save_at_controller_cut(path, mutation, monkeypatch):
    client = interface._maybe_create_tq_client()
    original = client.save_controller_checkpoint
    cut, release = Event(), Event()

    def checkpoint(controller_path):
        original(controller_path)
        cut.set()
        assert release.wait(30), "Storage checkpoint cut was not released"

    with monkeypatch.context() as patches:
        patches.setattr(client, "save_controller_checkpoint", checkpoint)
        with ThreadPoolExecutor(max_workers=1) as executor:
            save = executor.submit(tq.save_checkpoint, path)
            try:
                assert cut.wait(30), "Controller checkpoint did not reach cut"
                mutation()
            finally:
                release.set()
            save.result(timeout=30)


def fresh_restore(config, path):
    tq.close()
    tq.init(config)
    tq.load_checkpoint(path)


def physical_state(path):
    interface._maybe_create_tq_client().save_storage_checkpoint(str(path))
    with (path / "simple_storage" / "storage_unit_info.json").open() as f:
        entries = json.load(f)
    states = []
    for entry in sorted(entries, key=lambda entry: entry["position"]):
        file = path / "simple_storage" / f"su_{entry['position']}_{entry['storage_unit_id']}.pkl"
        with file.open("rb") as f:
            states.append(pickle.load(f))
    return states


def test_mixed_snapshot_prunes_exact_fields_and_indexes(services, tmp_path, monkeypatch):
    put("promptless-reader", partition="other-consumer", value=7)
    put("finished", value=2)
    put("field-owner", field="late-field", value=9)
    put("validation", partition="val", value=3)
    controller = interface._TQ_CONTROLLER
    ray.get(controller.create_partition.remote("index-only"))
    path = tmp_path / "checkpoint"

    def post_cut():
        put("finished", field="late-field", value=99)
        put("post-cut", field="old-only", value=88)

    save_at_controller_cut(path, post_cut, monkeypatch)
    with (path / "controller_state.pkl").open("rb") as f:
        saved = pickle.load(f)
    fresh_restore(services, path)
    assert tq.kv_batch_get(keys=["promptless-reader"], partition_id="other-consumer")["value"].unbind()[0].item() == 7
    assert tq.kv_batch_get(keys=["validation"], partition_id="val")["value"].unbind()[0].item() == 3
    assert tq.kv_batch_get(keys=["finished"], partition_id="train")["value"].unbind()[0].item() == 2
    states = physical_state(tmp_path / "restored")
    for position, state in enumerate(states):
        expected = {}
        for partition in saved["partitions"].values():
            for field, meta in partition.field_metadata.items():
                indexes = {index for index in meta.global_indexes if index % 2 == position}
                if indexes:
                    expected.setdefault(field, set()).update(indexes)
        assert {field: set(values) for field, values in state["field_data"].items()} == expected
        assert state["active_keys"] == set().union(*expected.values())
    put("fresh", value=4)
    for state in physical_state(tmp_path / "reallocated"):
        assert "old-only" not in state["field_data"]
        values = state["field_data"].get("late-field", {})
        expected_indexes = saved["partitions"]["train"].field_metadata["late-field"].global_indexes
        assert set(values).issubset(expected_indexes)


def test_missing_produced_field_refuses_fresh_restore(services, tmp_path, monkeypatch):
    put("required")
    path = tmp_path / "checkpoint"
    save_at_controller_cut(path, lambda: tq.kv_clear(keys=["required"], partition_id="train"), monkeypatch)
    tq.close()
    tq.init(services)
    with pytest.raises(RuntimeError, match="Missing produced checkpoint payload.*partition='train'.*key='required'"):
        tq.load_checkpoint(path)
    assert ray.get(interface._TQ_CONTROLLER.list_partitions.remote()) == []


def test_reused_index_and_empty_field_mapping_restore(services, tmp_path):
    put("old", field="cleared")
    tq.kv_clear(keys=["old"], partition_id="train")
    put("new", value=5)
    path = tmp_path / "checkpoint"
    tq.save_checkpoint(path)
    fresh_restore(services, path)
    assert tq.kv_batch_get(keys=["new"], partition_id="train")["value"].unbind()[0].item() == 5


@pytest.mark.parametrize("positions", [[0, 0], [-1, 1]])
def test_manifest_requires_exact_shard_positions(services, tmp_path, positions):
    put("saved")
    path = tmp_path / "checkpoint"
    tq.save_checkpoint(path)
    manifest = path / "simple_storage" / "storage_unit_info.json"
    entries = json.loads(manifest.read_text())
    for entry, position in zip(entries, positions, strict=True):
        entry["position"] = position
    manifest.write_text(json.dumps(entries))
    with pytest.raises(ValueError, match="positions must cover each shard exactly once"):
        tq.load_checkpoint(path)


@pytest.mark.asyncio
async def test_failed_load_waits_for_admitted_shards(services, tmp_path, monkeypatch):
    put("missing")
    put("retained")
    path = tmp_path / "checkpoint"
    tq.save_checkpoint(path)
    manifest = json.loads((path / "simple_storage" / "storage_unit_info.json").read_text())
    first = manifest[0]
    (path / "simple_storage" / f"su_0_{first['storage_unit_id']}.pkl").unlink()
    tq.close()
    tq.init(services)
    manager = interface._maybe_create_tq_client().storage_manager
    original = manager._load_single_storage_unit
    entered, failed, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    second_id = list(manager.storage_unit_infos)[1]

    async def held(*args, **kwargs):
        try:
            result = await original(*args, **kwargs)
        except RuntimeError:
            failed.set()
            raise
        if kwargs["target_storage_unit"] == second_id:
            entered.set()
            await release.wait()
        return result

    monkeypatch.setattr(manager, "_load_single_storage_unit", held)
    load = asyncio.create_task(manager.load_checkpoint(str(path)))
    try:
        await asyncio.wait_for(entered.wait(), 30)
        await asyncio.wait_for(failed.wait(), 30)
        await asyncio.sleep(0)
        assert not load.done(), "Restore returned before the other admitted shard settled"
    finally:
        release.set()
        with pytest.raises(RuntimeError, match="No such file"):
            await load


@pytest.mark.asyncio
async def test_cancelled_load_waits_through_repeated_cancellation(services, tmp_path, monkeypatch):
    put("first")
    put("second")
    path = tmp_path / "checkpoint"
    tq.save_checkpoint(path)
    manager = interface._maybe_create_tq_client().storage_manager
    original = manager._load_single_storage_unit
    entered, release, abandoned = asyncio.Event(), asyncio.Event(), asyncio.Event()
    completed = 0

    async def held(*args, **kwargs):
        nonlocal completed
        result = await original(*args, **kwargs)
        completed += 1
        if completed == 2:
            entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            abandoned.set()
            raise
        return result

    monkeypatch.setattr(manager, "_load_single_storage_unit", held)
    load = asyncio.create_task(manager.load_checkpoint(str(path)))
    try:
        await asyncio.wait_for(entered.wait(), 30)
        for _ in range(2):
            load.cancel()
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(abandoned.wait(), 0.05)
            assert not load.done(), "Cancellation abandoned admitted shard operations"
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await load


def test_wrongly_routed_produced_payload_refuses_restore(services, tmp_path):
    put("first", field="first-field")
    put("second", field="second-field")
    path = tmp_path / "checkpoint"
    tq.save_checkpoint(path)
    manifest = json.loads((path / "simple_storage" / "storage_unit_info.json").read_text())
    files = [path / "simple_storage" / f"su_{entry['position']}_{entry['storage_unit_id']}.pkl" for entry in manifest]
    first, second = [file.read_bytes() for file in files]
    files[0].write_bytes(second)
    files[1].write_bytes(first)
    tq.close()
    tq.init(services)
    with pytest.raises(RuntimeError, match="Missing produced checkpoint payload"):
        tq.load_checkpoint(path)


def test_unproduced_clearing_field_may_be_absent(services, tmp_path):
    put("clearing")
    client = interface._maybe_create_tq_client()
    meta = client.kv_retrieve_meta(keys=["clearing"], partition_id="train", create=False)
    ray.get(interface._TQ_CONTROLLER.mark_clearing.remote(meta.global_indexes, meta.partition_ids))
    client._run_coroutine(client.storage_manager.clear_data(meta))
    path = tmp_path / "checkpoint"
    tq.save_checkpoint(path)
    fresh_restore(services, path)
    snapshot = ray.get(interface._TQ_CONTROLLER.get_partition_snapshot.remote("train"))
    assert "clearing" in snapshot.keys_mapping
    assert not snapshot.production_status.any()
    assert all(not state["field_data"] and not state["active_keys"] for state in physical_state(tmp_path / "empty"))
