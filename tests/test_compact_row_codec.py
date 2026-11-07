# Copyright 2026 Zyphra Technologies Inc.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Stored rows use compact physical bytes; gets and checkpoint readers see the original rows."""

import io
import pickle

import pytest
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
from transfer_queue.utils.route_codec import Arange, NarrowInts, NibbleRoutes, PackedRoutes, pack_route_fields
from transfer_queue.utils.zmq_utils import ZMQMessage, ZMQRequestType


def unit():
    actor = object.__new__(SimpleStorageUnit.__ray_metadata__.modified_class)
    actor.storage_unit_id, actor.storage_unit_size = "codec-test", None
    actor.storage_data = StorageUnitData(None)
    return actor


def session(lengths, *, sentinel=False):
    """Rows shaped like an agent session: token ids, masks, text positions, log probs and R3 routes."""
    rows = []
    for prompt, response in lengths:
        total = prompt + response
        ids = torch.randint(0, 150_000, (total,))
        mask = torch.randint(0, 2, (response,))
        routes = torch.randint(0, 16, (total, 40, 1), dtype=torch.int16)
        if sentinel:
            routes[0, 0, 0] = -1
        rows.append(
            {
                "prompts": ids[:prompt].clone(),
                "responses": ids[prompt:].clone(),
                "input_ids": ids,
                "position_ids": torch.arange(total),
                "response_mask": mask,
                "loss_mask": mask.clone(),
                "rollout_log_probs": torch.randn(response),
                "routed_experts": routes,
                "wide": torch.full((3,), 1 << 40),
            }
        )
    fields = {k: torch.nested.as_nested_tensor([r[k] for r in rows], layout=torch.jagged) for k in rows[0]}
    return TensorDict(fields, batch_size=[len(rows)]), rows


def put(actor, data, keys):
    selected = {
        name: AsyncSimpleStorageManager._select_by_positions(data[name], list(range(len(keys)))) for name in data.keys()
    }
    request = ZMQMessage.create(
        ZMQRequestType.PUT_DATA, "test", {"global_indexes": keys, "data": pack_route_fields(selected)}
    )
    frames = [bytes(memoryview(frame)) for frame in request.serialize()]
    assert actor._handle_put(ZMQMessage.deserialize(frames, retain_packed_routes=True)).request_type == (
        ZMQRequestType.PUT_DATA_RESPONSE
    )


def assert_same(actual, expected):
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert actual.numpy().tobytes() == expected.numpy().tobytes()


@pytest.mark.parametrize("sentinel", [False, True], ids=["routes-0-15", "routes-with-minus-1"])
def test_put_stores_compact_rows_and_get_restores_them_exactly(sentinel):
    data, rows = session([(300, 700), (450, 1200)], sentinel=sentinel)
    actor = unit()
    put(actor, data, [0, 1])
    stored = actor.storage_data.field_data
    assert isinstance(stored["input_ids"][0], NarrowInts) and stored["input_ids"][0].data.dtype == torch.int32
    assert isinstance(stored["response_mask"][0], NarrowInts) and stored["response_mask"][0].data.dtype == torch.uint8
    assert isinstance(stored["position_ids"][0], Arange)
    assert type(stored["wide"][0]) is torch.Tensor and stored["wide"][0].dtype == torch.int64
    routes = stored["routed_experts"][0]
    assert isinstance(routes, PackedRoutes if sentinel else NibbleRoutes)
    got = actor.storage_data.get_data(list(rows[0]), [0, 1])
    for index, row in enumerate(rows):
        for name, expected in row.items():
            assert_same(got[name][index], expected)
    packed = actor.storage_data.get_data(["routed_experts"], [1], packed_routes=True)["routed_experts"][0]
    assert isinstance(packed, PackedRoutes)
    assert_same(packed.restore(), rows[1]["routed_experts"])


def test_compact_rows_hold_less_than_half_of_their_logical_bytes():
    data, rows = session([(2000, 5000)])
    actor = unit()
    put(actor, data, [0])
    logical = sum(value.numel() * value.element_size() for value in rows[0].values())
    held = 0
    for field in actor.storage_data.field_data.values():
        value = field[0]
        tensor = getattr(value, "data", value)
        held += tensor.numel() * tensor.element_size() if isinstance(tensor, torch.Tensor) else 0
    assert held < 0.45 * logical
    assert actor.storage_data.payload_bytes < 0.5 * logical


def test_checkpoints_hold_compact_rows_and_readers_restore_logical_rows(tmp_path):
    data, rows = session([(300, 700), (450, 1200)])
    actor = unit()
    put(actor, data, [0, 1])
    state = {"storage_unit_size": None, "field_data": actor.storage_data.field_data, "active_keys": {0, 1}}
    stream = io.BytesIO()
    _CheckpointPickler(stream, state).dump(state)
    ordinary = pickle.loads(stream.getvalue())["field_data"]
    for index, row in enumerate(rows):
        for name, expected in row.items():
            assert_same(ordinary[name][index], expected)
    path = tmp_path / "unit.ckpt"
    _write_storage_checkpoint(path, state)
    assert _read_storage_checkpoint(path)["field_data"]["input_ids"][1].dtype == torch.int64
    loaded = unit()
    request = ZMQMessage.create(ZMQRequestType.LOAD_STORAGE_CHECKPOINT, "test", {"path": str(path)})
    assert loaded._handle_load_checkpoint(request).body["success"]
    assert isinstance(loaded.storage_data.field_data["position_ids"][0], Arange)
    got = loaded.storage_data.get_data(list(rows[0]), [0, 1])
    for index, row in enumerate(rows):
        for name, expected in row.items():
            assert_same(got[name][index], expected)
