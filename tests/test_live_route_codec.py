# Copyright 2026 Zyphra Technologies Inc.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Public wire/storage roundtrips keep logical fields while retaining half-width routes."""

import pickle
import subprocess
import sys

import pytest
import torch
from tensordict import NonTensorStack, TensorDict

from transfer_queue.storage.simple_storage import SimpleStorageUnit, StorageUnitData
from transfer_queue.utils.route_codec import PackedRoutes, pack_route_fields
from transfer_queue.utils.zmq_utils import ZMQMessage, ZMQRequestType


def unit():
    actor = object.__new__(SimpleStorageUnit.__ray_metadata__.modified_class)
    actor.storage_unit_id, actor.storage_unit_size = "live-route-test", 8
    actor.storage_data = StorageUnitData(8)
    return actor


def put(fields):
    return ZMQMessage.create(ZMQRequestType.PUT_DATA, "test", {"global_indexes": [0, 1], "data": fields})


def get():
    return ZMQMessage.create(
        ZMQRequestType.GET_DATA, "test", {"global_indexes": [0, 1], "fields": ["routed_experts", "input_ids"]}
    )


@pytest.mark.parametrize("layout", ["list", "dense", "jagged"])
def test_real_wire_put_physical_bytes_get_and_restore(tmp_path, layout):
    rows = [torch.arange(320, dtype=torch.int16).reshape(8, 40, 1) % 16 for _ in range(2)]
    routes = (
        rows
        if layout == "list"
        else (torch.stack(rows) if layout == "dense" else torch.nested.as_nested_tensor(rows, layout=torch.jagged))
    )
    tokens = torch.arange(16).reshape(2, 8)
    original = {"routed_experts": routes, "input_ids": tokens}
    old_frames = put(original).serialize()
    frames = put(pack_route_fields(original)).serialize()
    assert sum(memoryview(x).nbytes for x in frames[1:]) < sum(memoryview(x).nbytes for x in old_frames[1:])
    request = ZMQMessage.deserialize(frames, retain_packed_routes=True)
    actor = unit()
    assert actor._handle_put(request).request_type == ZMQRequestType.PUT_DATA_RESPONSE
    physical = actor.storage_data.field_data["routed_experts"]
    assert all(isinstance(value, PackedRoutes) and value.data.dtype == torch.uint8 for value in physical.values())
    assert sum(value.data.nbytes for value in physical.values()) == sum(value.nbytes for value in rows) // 2
    received = ZMQMessage.deserialize(actor._handle_get(get()).serialize()).body["data"]
    for index, expected in enumerate(rows):
        assert received["routed_experts"][index].dtype == torch.int16
        assert torch.equal(received["routed_experts"][index], expected)
        assert torch.equal(received["input_ids"][index], tokens[index])
    path = tmp_path / "storage.pkl"
    save = ZMQMessage.create(ZMQRequestType.SAVE_STORAGE_CHECKPOINT, "test", {"path": str(path)})
    assert actor._handle_save_checkpoint(save).body["failure"] is None
    restored = unit()
    assert restored._handle_load_checkpoint(save).body["success"]
    again = ZMQMessage.deserialize(restored._handle_get(get()).serialize()).body["data"]
    assert all(torch.equal(a, b) for a, b in zip(again["routed_experts"], received["routed_experts"], strict=True))
    assert all(isinstance(value, PackedRoutes) for value in restored.storage_data.field_data["routed_experts"].values())


@pytest.mark.parametrize("values", [[0, 255], [-1, 0], [-1, 254], [-1, 255], [0, 256], [-32768, 32767], []])
def test_range_guards_and_mixed_old_wire(values):
    route = torch.tensor(values, dtype=torch.int16)
    fields = pack_route_fields({"routed_experts": [route]})
    physical = fields["routed_experts"][0]
    eligible = all(0 <= value <= 255 for value in values) or (min(values, default=0) == -1 and max(values) <= 254)
    assert isinstance(physical, PackedRoutes) == eligible
    message = ZMQMessage.create(ZMQRequestType.PUT_DATA, "test", {"data": fields})
    restored = ZMQMessage.deserialize(message.serialize()).body["data"]["routed_experts"][0]
    assert restored.dtype == route.dtype and torch.equal(restored, route)
    legacy = ZMQMessage.create(ZMQRequestType.PUT_DATA, "test", {"data": {"routed_experts": [route]}})
    old = ZMQMessage.deserialize(legacy.serialize(), retain_packed_routes=True).body["data"]["routed_experts"][0]
    assert old.dtype == torch.int16 and torch.equal(old, route)


def test_cross_field_and_overlapping_aliases_keep_original_values():
    route = torch.arange(16, dtype=torch.int16)
    fields = {"routed_experts": [route], "alias": [route]}
    packed = pack_route_fields(fields)
    assert packed["routed_experts"][0] is packed["alias"][0] is route
    fields = {"routed_experts": [route[:10], route[5:]], "input_ids": [torch.arange(2)]}
    packed = pack_route_fields(fields)
    assert all(type(value) is torch.Tensor for value in packed["routed_experts"])
    packed["routed_experts"][0][5] = 98
    assert packed["routed_experts"][1][0] == 98


def test_known_tensordict_metadata_allows_accounted_async_cut():
    rows = NonTensorStack({"field": "one"}, {"field": "two"})
    actor = unit()
    request = put({"routed_experts": [torch.arange(16, dtype=torch.int16)] * 2, "metadata": rows})
    decoded = ZMQMessage.deserialize(request.serialize(), retain_packed_routes=True)
    assert actor._handle_put(decoded).request_type == ZMQRequestType.PUT_DATA_RESPONSE
    assert all(isinstance(value, TensorDict) for value in actor.storage_data.field_data["metadata"].values())
    snapshot, size = actor.storage_data.checkpoint_snapshot({0, 1})
    assert size is not None and size > 32
    ordinary = pickle.loads(pickle.dumps(snapshot))
    assert ordinary["active_keys"] == {0, 1}


@pytest.mark.parametrize("dtype", [torch.uint8, torch.int32, torch.float32])
def test_other_route_dtypes_are_unchanged(dtype):
    value = torch.arange(16, dtype=dtype)
    assert pack_route_fields({"routed_experts": [value]})["routed_experts"][0] is value


@pytest.mark.parametrize("offset", [-1, 0])
def test_packed_checkpoint_uses_ordinary_torch_reader(tmp_path, offset):
    value = PackedRoutes(torch.tensor([0, 1, 254], dtype=torch.uint8), offset=offset)
    path = tmp_path / "route.pkl"
    path.write_bytes(pickle.dumps(value))
    code = (
        "import pickle,sys,torch;v=pickle.load(open(sys.argv[1],'rb'));"
        "assert v.dtype==torch.int16;"
        "assert torch.equal(v,torch.tensor([0,1,254],dtype=torch.int16)+int(sys.argv[2]))"
    )
    subprocess.run([sys.executable, "-I", "-c", code, str(path), str(offset)], check=True, timeout=30)


@pytest.mark.parametrize("dtype,offset", [(torch.int16, 0), (torch.uint8, 1)])
def test_invalid_physical_encoding_is_refused(dtype, offset):
    with pytest.raises(ValueError):
        PackedRoutes(torch.tensor([0, 1], dtype=dtype), offset=offset)
