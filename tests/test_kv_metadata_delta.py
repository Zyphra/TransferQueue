"""Public metadata cursors observe external writers and reset without losing deletions."""

import asyncio
import tempfile
from collections import deque
from pathlib import Path
from threading import RLock

import pytest
import ray
from omegaconf import OmegaConf

import transfer_queue as tq
from transfer_queue.controller import DataPartitionStatus, TransferQueueController


@pytest.fixture(scope="module")
def runtime():
    with tempfile.TemporaryDirectory(prefix="tq-delta-") as tmp:
        ray.init(
            address="local",
            num_cpus=4,
            include_dashboard=False,
            _temp_dir=tmp,
            object_store_memory=128 * 1024**2,
            _memory=1024**3,
        )
        tq.init(
            OmegaConf.create(
                {
                    "backend": {
                        "storage_backend": "SimpleStorage",
                        "SimpleStorage": {
                            "total_storage_size": 100000,
                            "num_data_storage_units": 1,
                        },
                    }
                }
            )
        )
        try:
            yield ray.get_actor("TransferQueueController", namespace="transfer_queue")
        finally:
            tq.close()
            ray.shutdown()


@pytest.fixture
def controller(runtime):
    yield runtime
    for partition in ray.get(runtime.list_partitions.remote()):
        ray.get(runtime.clear_partition.remote(partition))


def write(controller, partition, tags):
    metadata = ray.get(controller.kv_retrieve_meta.remote(list(tags), partition, create=True))
    values = dict(zip(metadata.global_indexes, tags.values(), strict=True))
    ray.get(controller.set_custom_meta.remote({partition: values}))
    return dict(zip(tags, metadata.global_indexes, strict=True))


def apply(view, delta):
    if delta["reset"]:
        view.clear()
    for partition, keys in delta["deleted"].items():
        for key in keys:
            view.get(partition, {}).pop(key, None)
    for partition, items in delta["upserts"].items():
        view.setdefault(partition, {}).update(items)


@pytest.mark.parametrize("use_async", [False, True])
def test_public_cursor_observes_external_tag_replacement_and_unchanged_polls(controller, use_async):
    poll = lambda cursor=None: asyncio.run(tq.async_kv_list_delta(cursor)) if use_async else tq.kv_list_delta(cursor)
    indexes = write(controller, "train", {"one": {"status": "running", "old": True}})
    first = poll()
    assert first["reset"] and first["upserts"] == tq.kv_list()
    unchanged = poll(first["cursor"])
    assert unchanged == {"cursor": first["cursor"], "reset": False, "upserts": {}, "deleted": {}}
    # A separate Ray control-plane request changes tags; no consumer-local epoch changes.
    ray.get(controller.set_custom_meta.remote({"train": {indexes["one"]: {"status": "finished"}}}))
    delta = poll(first["cursor"])
    assert not delta["reset"] and delta["upserts"] == {"train": {"one": {"status": "finished"}}}
    assert first["upserts"]["train"]["one"] == {"status": "running", "old": True}


def test_delete_then_recreate_is_applied_in_order_without_losing_other_partitions(controller):
    indexes = write(controller, "train", {"one": {"v": 1}, "two": {"v": 2}})
    write(controller, "val", {"kept": {"v": 3}})
    first = tq.kv_list_delta()
    view = {}
    apply(view, first)
    ray.get(controller.clear_meta.remote(global_indexes=[indexes["one"]], partition_ids=["train"]))
    write(controller, "train", {"between": {"v": 5}})
    write(controller, "train", {"one": {"v": 4}})
    delta = tq.kv_list_delta(first["cursor"])
    assert delta["deleted"] == {"train": ["one"]}
    assert delta["upserts"] == {"train": {"between": {"v": 5}, "one": {"v": 4}}}
    apply(view, delta)
    assert view == tq.kv_list() and list(view["train"]) == ["two", "between", "one"]


def test_partition_replacement_and_restore_invalidate_old_cursors(controller, tmp_path):
    write(controller, "train", {"saved": {"v": 1}})
    saved = tq.kv_list_delta()
    path = str(tmp_path / "controller.pkl")
    ray.get(controller.save_checkpoint.remote(path))
    ray.get(controller.clear_partition.remote("train"))
    write(controller, "train", {"new": {"v": 2}})
    changed = tq.kv_list_delta(saved["cursor"])
    assert changed["reset"] and changed["upserts"] == {"train": {"new": {"v": 2}}}
    ray.get(controller.load_checkpoint.remote(path))
    restored = tq.kv_list_delta(changed["cursor"])
    assert restored["reset"] and restored["upserts"] == saved["upserts"]
    assert restored["cursor"].split(":")[0] not in {
        saved["cursor"].split(":")[0],
        changed["cursor"].split(":")[0],
    }
    # The journal and cursor are not stored in the checkpoint.
    assert Path(path).is_file()


def test_bounded_journal_overflow_returns_a_full_view_instead_of_missing_keys(controller):
    write(controller, "train", {"seed": {"v": 0}})
    cursor = tq.kv_list_delta()["cursor"]
    keys = [f"new-{i}" for i in range(65537)]
    ray.get(controller.kv_retrieve_meta.remote(keys, "train", create=True))
    delta = tq.kv_list_delta(cursor)
    assert delta["reset"] and len(delta["upserts"]["train"]) == 65538
    assert delta["upserts"] == tq.kv_list()


def test_delta_reply_owns_nested_tags_before_transport_serialization():
    impl = TransferQueueController.__ray_metadata__.modified_class
    controller = object.__new__(impl)
    controller._kv_lock = RLock()
    controller._kv_changes = deque(maxlen=65536)
    controller._reset_kv_delta()
    partition = DataPartitionStatus(partition_id="train")
    partition.keys_mapping = {"one": 0}
    partition.revert_keys_mapping = {0: "one"}
    partition.custom_meta = {0: {"nested": {"attempts": [1]}}}
    controller.partitions = {"train": partition}
    first = controller.kv_list_delta()
    partition.custom_meta[0]["nested"]["attempts"].append(2)
    controller._record_kv_changes("train", ["one"])
    delta = controller.kv_list_delta(first["cursor"])
    partition.custom_meta[0]["nested"]["attempts"].append(3)
    assert first["upserts"]["train"]["one"]["nested"]["attempts"] == [1]
    assert delta["upserts"]["train"]["one"]["nested"]["attempts"] == [1, 2]
