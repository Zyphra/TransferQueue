# Copyright 2026 Zyphra Technologies Inc.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Stored rows own exactly their bytes, in memory and in checkpoints."""

import io

import torch
from tensordict import TensorDict

from transfer_queue.storage.managers.simple_storage_manager import AsyncSimpleStorageManager
from transfer_queue.storage.simple_storage import (
    SimpleStorageUnit,
    StorageUnitData,
    _CheckpointPickler,
    _read_storage_checkpoint,
    _write_storage_checkpoint,
)
from transfer_queue.utils.route_codec import PackedRoutes
from transfer_queue.utils.zmq_utils import ZMQMessage, ZMQRequestType

ROWS, TOKENS = 64, 2000
# Irregular hash-routed positions take the manager's index_select path: one dense frame per unit.
POSITIONS = [0, 3, 4, 9, 10, 17, 30, 31, 40, 47, 50, 63]


def unit():
    actor = object.__new__(SimpleStorageUnit.__ray_metadata__.modified_class)
    actor.storage_unit_id, actor.storage_unit_size = "ownership-test", 1024
    actor.storage_data = StorageUnitData(1024)
    return actor


def batch():
    torch.manual_seed(0)
    return TensorDict(
        {
            # Values beyond int32 keep int64 storage, so the codec leaves these rows as plain tensors.
            "input_ids": torch.randint(1 << 40, (1 << 40) + 50_000, (ROWS, TOKENS)),
            "log_probs": torch.randn(ROWS, TOKENS),
            "routed_experts": torch.randint(0, 64, (ROWS, TOKENS, 4), dtype=torch.int16),
        },
        batch_size=[ROWS],
    )


def put_through_transport(actor, data):
    """The real client-to-unit path: per-unit selection, zero-copy frames, unit-side decode and put."""
    selected = {name: AsyncSimpleStorageManager._select_by_positions(data[name], POSITIONS) for name in data.keys()}
    request = ZMQMessage.create(ZMQRequestType.PUT_DATA, "test", {"global_indexes": POSITIONS, "data": selected})
    frames = [bytes(memoryview(frame)) for frame in request.serialize()]
    received = ZMQMessage.deserialize(frames, retain_packed_routes=True)
    assert actor._handle_put(received).request_type == ZMQRequestType.PUT_DATA_RESPONSE


def stored_tensor(value):
    return value.data if isinstance(value, PackedRoutes) else value


def logical_bytes(rows):
    return sum(stored_tensor(value).numel() * stored_tensor(value).element_size() for value in rows.values())


def pickled_bytes(field_data):
    stream = io.BytesIO()
    state = {"field_data": field_data, "active_keys": set().union(*field_data.values())}
    _CheckpointPickler(stream, state).dump(state)
    return len(stream.getvalue())


def test_batched_put_stores_owning_rows_with_identical_bytes():
    data, actor = batch(), unit()
    put_through_transport(actor, data)
    for name in ("input_ids", "log_probs"):
        for key in POSITIONS:
            value = actor.storage_data.field_data[name][key]
            assert value.untyped_storage().nbytes() == value.numel() * value.element_size()
            assert value.numpy().tobytes() == data[name][key].numpy().tobytes()
    restored = actor.storage_data.get_data(["routed_experts"], POSITIONS)["routed_experts"]
    assert all(torch.equal(row, data["routed_experts"][key]) for row, key in zip(restored, POSITIONS, strict=True))


def test_owning_rows_are_stored_without_a_copy():
    rows = [torch.arange(TOKENS) * 3 + (1 << 40) + index for index in range(3)]
    data = StorageUnitData(8)
    data.put_data({"input_ids": rows}, [0, 1, 2])
    assert all(data.field_data["input_ids"][index] is row for index, row in enumerate(rows))


def test_drained_rows_release_siblings_and_checkpoint_only_their_bytes(tmp_path):
    data, actor = batch(), unit()
    put_through_transport(actor, data)
    fields = actor.storage_data.field_data
    # The same rows held as views of their dense frame, as stored before rows owned their bytes.
    frames = {name: data[name][POSITIONS].clone() for name in ("input_ids", "log_probs")}
    views = {name: dict(zip(POSITIONS, frames[name], strict=True)) for name in frames}
    logical = sum(logical_bytes(fields[name]) for name in views)
    assert pickled_bytes(views) > 10 * logical
    assert pickled_bytes({name: fields[name] for name in views}) < logical + 128 * 1024

    actor.storage_data.clear(POSITIONS[1:])
    survivor = POSITIONS[0]
    one_row = sum(stored_tensor(fields[name][survivor]).untyped_storage().nbytes() for name in fields)
    assert one_row == sum(logical_bytes({survivor: fields[name][survivor]}) for name in fields)

    path = tmp_path / "unit.ckpt"
    _write_storage_checkpoint(path, {"storage_unit_size": 1024, "field_data": fields, "active_keys": {survivor}})
    restored = _read_storage_checkpoint(path)["field_data"]
    for name in ("input_ids", "log_probs"):
        assert restored[name][survivor].numpy().tobytes() == data[name][survivor].numpy().tobytes()


def test_restore_compacts_rows_saved_as_batch_views(tmp_path):
    data = batch()
    frame = data["input_ids"][POSITIONS].clone()
    path = tmp_path / "views.ckpt"
    state = {
        "storage_unit_size": 1024,
        "field_data": {"input_ids": dict(zip(POSITIONS, frame, strict=True))},
        "active_keys": set(POSITIONS),
    }
    _write_storage_checkpoint(path, state)
    actor = unit()
    request = ZMQMessage.create(ZMQRequestType.LOAD_STORAGE_CHECKPOINT, "test", {"path": str(path)})
    assert actor._handle_load_checkpoint(request).body["success"]
    for key in POSITIONS:
        value = actor.storage_data.field_data["input_ids"][key]
        assert value.untyped_storage().nbytes() == value.numel() * value.element_size()
        assert torch.equal(value, data["input_ids"][key])
