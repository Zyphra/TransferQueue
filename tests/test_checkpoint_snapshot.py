# Copyright 2026 Zyphra Technologies Inc.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Payload ownership and sampler detachment for bounded checkpoint capture."""

import pickle
from types import SimpleNamespace

import pytest
import torch

from transfer_queue.controller import TransferQueueController
from transfer_queue.storage.simple_storage import StorageUnitData


def test_snapshot_retains_original_values_and_full_tensor_backing_store():
    data = StorageUnitData(storage_size=4)
    backing = torch.arange(4096)
    data.put_data({"tokens": [backing[:2]], "nested": [{"original": [1, 2]}]}, [0])
    snapshot, size = data.checkpoint_snapshot({0})
    assert size >= backing.untyped_storage().nbytes()
    data.put_data({"tokens": [torch.tensor([99])], "nested": [{"changed": True}]}, [0])
    data.clear([0])
    data.put_data({"tokens": [torch.tensor([100])]}, [0])
    assert torch.equal(snapshot["field_data"]["tokens"][0], torch.tensor([0, 1]))
    assert snapshot["field_data"]["nested"][0] == {"original": [1, 2]}
    assert snapshot["active_keys"] == {0}


def test_unknown_payload_ownership_selects_synchronous_fallback():
    data = StorageUnitData(storage_size=4)
    data.put_data({"opaque": [SimpleNamespace(value=1)]}, [0])
    _, size = data.checkpoint_snapshot({0})
    assert size is None


def test_controller_freezes_sampler_before_serialization(tmp_path, monkeypatch):
    impl = TransferQueueController.__ray_metadata__.modified_class
    controller = object.__new__(impl)
    controller.controller_id, controller.partitions = "snapshot-control", {}
    controller.index_manager = SimpleNamespace(
        partition_to_indexes={},
        reusable_indexes=[],
        global_index_counter=0,
        allocated_indexes=set(),
    )
    sampler_state = {"cursor": [3]}
    controller.sampler = SimpleNamespace(save_checkpoint=lambda: sampler_state)
    original = pickle.dump

    def change_live_sampler(state, stream, **kwargs):
        sampler_state["cursor"][0] = 9
        return original(state, stream, **kwargs)

    monkeypatch.setattr(pickle, "dump", change_live_sampler)
    path = tmp_path / "controller.pkl"
    assert controller.save_checkpoint(str(path), return_frontier=True) == []
    with path.open("rb") as stream:
        assert pickle.load(stream)["sampler"] == {"cursor": [3]}
    assert sampler_state == {"cursor": [9]}


@pytest.fixture(scope="module")
def three_cpu_ray(tmp_path_factory):
    import ray

    assert not ray.is_initialized()
    ray.init(
        num_cpus=3,
        num_gpus=0,
        include_dashboard=False,
        object_store_memory=128 * 1024 * 1024,
        _node_ip_address="127.0.0.1",
        _temp_dir=str(tmp_path_factory.mktemp("two-su-ray")),
    )
    try:
        yield
    finally:
        ray.shutdown()


@pytest.fixture
def two_units(three_cpu_ray):
    from omegaconf import OmegaConf

    import transfer_queue as tq

    tq.init(
        OmegaConf.create(
            {
                "backend": {
                    "storage_backend": "SimpleStorage",
                    "SimpleStorage": {
                        "total_storage_size": 64,
                        "num_data_storage_units": 2,
                    },
                },
                "metrics": {"enabled": False},
            }
        )
    )
    try:
        yield tq
    finally:
        tq.close()


def _wait_file(path):
    import time

    deadline = time.monotonic() + 10
    while not path.exists():
        assert time.monotonic() < deadline, str(path)
        time.sleep(0.01)


def _install_unit_control(tq, key, value, tmp_path, *, capture=False, prepare_fail=False, write_fail=False):
    from functools import partial

    from tensordict import TensorDict

    def install(data, *, capture, prepare_fail, write_fail, root):
        import pickle
        import threading
        import time
        from pathlib import Path

        from transfer_queue.storage.simple_storage import StorageUnitData

        root = Path(root)

        def wait(name):
            (root / (name + "-entered")).touch()
            deadline = time.monotonic() + 20
            while not (root / (name + "-release")).exists():
                if time.monotonic() >= deadline:
                    raise TimeoutError("owned two-unit checkpoint gate deadline")
                time.sleep(0.01)

        original_capture, original_dump = StorageUnitData.checkpoint_snapshot, pickle.dump

        def checkpoint_snapshot(storage, indexes):
            if capture:
                wait("capture")
                if prepare_fail:
                    raise RuntimeError("injected storage capture failure")
            snapshot = original_capture(storage, indexes)
            (root / "captured").touch()
            return snapshot

        def dump(state, stream, *args, **kwargs):
            if threading.current_thread().name == "StorageCheckpointWriter":
                wait("write")
                if write_fail:
                    raise OSError("injected storage write failure")
            return original_dump(state, stream, *args, **kwargs)

        StorageUnitData.checkpoint_snapshot, pickle.dump = checkpoint_snapshot, dump
        return data

    tmp_path.mkdir()
    meta = tq.get_client().kv_retrieve_meta([key], "two-su")
    tq.get_client().put(
        TensorDict({"tokens": value.unsqueeze(0)}, batch_size=[1]),
        metadata=meta,
        data_parser=partial(
            install, capture=capture, prepare_fail=prepare_fail, write_fail=write_fail, root=str(tmp_path)
        ),
    )
    manager = tq.get_client().storage_manager
    position = meta.global_indexes[0] % 2
    return position, list(manager.storage_unit_infos)[position]


def _unit_status(tq, checkpoint, unit):
    position, identity = unit
    manager = tq.get_client().storage_manager
    return tq.get_client()._run_coroutine(
        manager._save_single_storage_unit(
            str(checkpoint / "simple_storage" / f"su_{position}_{identity}.pkl"),
            target_storage_unit=identity,
            phase="status",
        )
    )["done"]


def _wait_unit(tq, checkpoint, unit, *, failed=False):
    import time

    deadline = time.monotonic() + 10
    while True:
        try:
            done = _unit_status(tq, checkpoint, unit)
        except RuntimeError as error:
            assert failed and "injected storage write failure" in str(error)
            return
        if done:
            assert not failed
            return
        assert time.monotonic() < deadline
        time.sleep(0.01)


@pytest.mark.parametrize("failure", [None, "write", "prepare"], ids=["success", "write-failure", "prepare-failure"])
def test_two_unit_capture_frontier_and_partial_failure(two_units, tmp_path, failure):
    import hashlib
    from concurrent.futures import ThreadPoolExecutor

    tq = two_units
    values = [torch.arange(12), torch.arange(17) + 100]
    for index, value in enumerate(values):
        tq.kv_put(key=f"row{index}", partition_id="two-su", fields={"tokens": value}, tag={"original": index})
    prior = tmp_path / "prior"
    tq.save_checkpoint(prior)
    original_files = {
        str(p.relative_to(prior)): hashlib.sha256(p.read_bytes()).hexdigest() for p in prior.rglob("*") if p.is_file()
    }
    gates = [tmp_path / "unit0", tmp_path / "unit1"]
    units = [
        _install_unit_control(tq, "row0", values[0], gates[0], write_fail=failure == "write"),
        _install_unit_control(tq, "row1", values[1], gates[1], capture=True, prepare_fail=failure == "prepare"),
    ]
    assert {unit[0] for unit in units} == {0, 1}
    path, staging = tmp_path / "checkpoint", tmp_path / "checkpoint.tmp"
    with ThreadPoolExecutor(max_workers=1) as executor:
        capture = executor.submit(tq.save_checkpoint, path, async_storage=True, max_snapshot_bytes=1024 * 1024)
        try:
            _wait_file(gates[0] / "write-entered")
            _wait_file(gates[1] / "capture-entered")
            assert not capture.done(), "all capture acknowledgments must precede handoff"
            # This new-after-C row routes to the already captured, unblocked SU.
            tq.kv_put(key="after-C", partition_id="two-su", fields={"tokens": torch.tensor([999])})
            fresh = tq.get_client().kv_retrieve_meta(["after-C"], "two-su")
            assert fresh.global_indexes[0] % 2 == units[0][0]
            gates[1].joinpath("capture-release").touch()
            if failure == "prepare":
                with pytest.raises(RuntimeError, match="injected storage capture failure") as first:
                    capture.result(timeout=5)
                assert not _unit_status(tq, staging, units[0]), "other writer must remain active"
            else:
                future = capture.result(timeout=5)
                assert future is not None
                _wait_file(gates[1] / "write-entered")
                assert not future.done()
                assert not _unit_status(tq, staging, units[1])
                if failure is None:
                    for index in range(2):
                        tq.kv_put(key=f"row{index}", partition_id="two-su", fields={"tokens": torch.tensor([-1])})
                    tq.kv_clear(keys=["row0", "row1"], partition_id="two-su")
                gates[0].joinpath("write-release").touch()
                _wait_unit(tq, staging, units[0], failed=failure == "write")
                if failure == "write":
                    with pytest.raises(RuntimeError, match="injected storage write failure") as first:
                        future.result(timeout=5)
                else:
                    assert not future.done(), "one durable SU cannot publish the checkpoint"
            assert not path.exists()
            assert staging.exists()
            assert not (staging / "metadata.json").exists()
            assert not (staging / "simple_storage" / "storage_unit_info.json").exists()
            if failure is not None:
                with pytest.raises(RuntimeError) as repeated:
                    tq.save_checkpoint(tmp_path / "rejected", async_storage=True, max_snapshot_bytes=1024 * 1024)
                assert repeated.value is first.value
                assert staging.exists(), "failed save must not delete another active writer's files"
                assert not (tmp_path / "rejected.tmp").exists()
        finally:
            for gate in gates:
                for name in ("capture-release", "write-release"):
                    gate.joinpath(name).touch()
            try:
                future = capture.result(timeout=10)
                if future is not None:
                    future.result(timeout=10)
            except RuntimeError as error:
                assert failure is not None and "injected storage" in str(error)
            if failure is not None:
                _wait_unit(tq, staging, units[0], failed=failure == "write")
                if failure != "prepare":
                    _wait_unit(tq, staging, units[1])
    assert original_files == {
        str(p.relative_to(prior)): hashlib.sha256(p.read_bytes()).hexdigest() for p in prior.rglob("*") if p.is_file()
    }
    if failure is None:
        assert (path / "metadata.json").is_file()
        assert (path / "simple_storage" / "storage_unit_info.json").is_file()
        tq.load_checkpoint(path)
        assert set(tq.kv_list("two-su")["two-su"]) == {"row0", "row1"}
        for index, value in enumerate(values):
            restored = tq.kv_batch_get(keys=[f"row{index}"], partition_id="two-su")["tokens"][0]
            assert torch.equal(restored, value)


def test_two_unit_mixed_budget_fallback_waits_for_detached_writer(two_units, tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    tq = two_units
    values = [torch.arange(12), torch.arange(3072)]
    gates = [tmp_path / "small", tmp_path / "large"]
    for index, value in enumerate(values):
        tq.kv_put(key=f"row{index}", partition_id="two-su", fields={"tokens": value})
    units = [_install_unit_control(tq, f"row{i}", value, gates[i]) for i, value in enumerate(values)]
    assert {unit[0] for unit in units} == {0, 1}
    path, staging = tmp_path / "checkpoint", tmp_path / "checkpoint.tmp"
    with ThreadPoolExecutor(max_workers=1) as executor:
        save = executor.submit(tq.save_checkpoint, path, async_storage=True, max_snapshot_bytes=48 * 1024)
        try:
            _wait_file(gates[0] / "write-entered")
            _wait_file(gates[1] / "captured")
            _wait_unit(tq, staging, units[1])
            assert not (gates[1] / "write-entered").exists(), "over-budget SU must write synchronously"
            assert not _unit_status(tq, staging, units[0])
            assert not save.done(), "mixed fallback must join the detached writer before returning"
            assert not path.exists()
            assert not (staging / "simple_storage" / "storage_unit_info.json").exists()
        finally:
            for gate in gates:
                gate.joinpath("write-release").touch()
            assert save.result(timeout=10) is None
    tq.load_checkpoint(path)
    for index, value in enumerate(values):
        assert torch.equal(tq.kv_batch_get(keys=[f"row{index}"], partition_id="two-su")["tokens"][0], value)
