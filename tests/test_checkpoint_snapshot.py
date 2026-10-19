# Copyright 2026 Zyphra Technologies Inc.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Payload ownership and sampler detachment for bounded checkpoint capture."""

import pickle
from types import SimpleNamespace

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
