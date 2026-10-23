# Copyright 2026 Zyphra Technologies Inc.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Checkpoint-only route packing, ordinary-reader compatibility and write cost."""

import io
import json
import os
import pickle
import subprocess
import sys
import tempfile
import time
from threading import Event

import numpy as np
import pytest
import torch

from transfer_queue.storage.simple_storage import (
    SimpleStorageUnit,
    StorageUnitData,
    _CheckpointPickler,
)
from transfer_queue.utils.zmq_utils import ZMQMessage, ZMQRequestType


def state_for(route, **fields):
    return {
        "storage_unit_id": "route-test",
        "storage_unit_size": 8,
        "field_data": {"routed_experts": {0: route}, **{name: {0: value} for name, value in fields.items()}},
        "active_keys": {0},
    }


def packed(state):
    stream = io.BytesIO()
    _CheckpointPickler(stream, state).dump(state)
    return stream.getvalue()


def unit():
    actor = object.__new__(SimpleStorageUnit.__ray_metadata__.modified_class)
    actor.storage_unit_id, actor.storage_unit_size = "route-test", 8
    actor.storage_data = StorageUnitData(8)
    return actor


def request(path, **body):
    return ZMQMessage.create(ZMQRequestType.SAVE_STORAGE_CHECKPOINT, "test", {"path": str(path), **body})


@pytest.mark.parametrize("values", [[0, 1, 15, 255], [], [-1, 0], [0, 256], [-32768, 32767]])
def test_exact_dtype_shape_values_and_range_fallback(values):
    route = torch.tensor(values, dtype=torch.int16).reshape(-1, 1, 1).clone()
    state = state_for(route, tokens=torch.tensor([19], dtype=torch.int64))
    payload = packed(state)
    restored = pickle.loads(payload)
    actual = restored["field_data"]["routed_experts"][0]
    assert actual.dtype == route.dtype and actual.shape == route.shape
    assert torch.equal(actual, route)
    assert torch.equal(restored["field_data"]["tokens"][0], state["field_data"]["tokens"][0])
    assert restored["active_keys"] == {0}
    assert (b"_rebuild_device_tensor_from_numpy" in payload) == all(0 <= value <= 255 for value in values)


@pytest.mark.parametrize(
    "kind", ["view", "full_view", "transpose", "alias", "nested_alias", "numpy_alias", "opaque", "float"]
)
def test_unsupported_or_shared_backing_uses_normal_pickle(kind):
    base = torch.arange(16, dtype=torch.int16).reshape(4, 4).clone()
    route, other = base, {}
    if kind == "view":
        route = base[:2]
    elif kind == "full_view":
        route = base.view(-1)
    elif kind == "transpose":
        route = base.T
    elif kind == "alias":
        other = {"alias": base.detach()}
    elif kind == "nested_alias":
        other = {"nested": {"alias": base.view(-1)}}
    elif kind == "numpy_alias":
        other = {"array": base.numpy()}
    elif kind == "opaque":
        other = {"dtype": torch.int16}
    elif kind == "float":
        route = base.float()
    state = state_for(route, **other)
    payload = packed(state)
    assert (b"_rebuild_device_tensor_from_numpy" in payload) == (kind in ("view", "full_view"))
    restored = pickle.loads(payload)["field_data"]
    assert torch.equal(restored["routed_experts"][0], route)
    if kind in ("alias", "nested_alias"):
        baseline = pickle.loads(pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL))["field_data"]
        alias = restored["alias"][0] if kind == "alias" else restored["nested"][0]["alias"]
        old_alias = baseline["alias"][0] if kind == "alias" else baseline["nested"][0]["alias"]
        assert torch.equal(alias, old_alias) and alias.shape == old_alias.shape and alias.dtype == old_alias.dtype
        assert (alias.untyped_storage().data_ptr() == restored["routed_experts"][0].untyped_storage().data_ptr()) == (
            old_alias.untyped_storage().data_ptr() == baseline["routed_experts"][0].untyped_storage().data_ptr()
        )


def test_same_tensor_object_keeps_identity_across_fields():
    route = torch.arange(16, dtype=torch.int16)
    state = state_for(route, duplicate=route)
    restored = pickle.loads(packed(state))["field_data"]
    ordinary = pickle.loads(pickle.dumps(state))["field_data"]
    assert restored["routed_experts"][0] is restored["duplicate"][0]
    restored["duplicate"][0][0] = ordinary["duplicate"][0][0] = 42
    assert torch.equal(restored["routed_experts"][0], ordinary["routed_experts"][0])


@pytest.mark.parametrize("overlap", [False, True])
def test_shared_route_spans_preserve_ordinary_alias_behavior(overlap):
    backing = torch.arange(32, dtype=torch.int16)
    rows = {0: backing[:16], 1: backing[8:24] if overlap else backing[16:]}
    state = state_for(rows[0])
    state["field_data"]["routed_experts"] = rows
    payload = packed(state)
    assert (b"_rebuild_device_tensor_from_numpy" in payload) != overlap
    actual, expected = pickle.loads(payload), pickle.loads(pickle.dumps(state))
    actual["field_data"]["routed_experts"][0].fill_(99)
    expected["field_data"]["routed_experts"][0].fill_(99)
    for index in rows:
        assert torch.equal(
            actual["field_data"]["routed_experts"][index], expected["field_data"]["routed_experts"][index]
        )


def test_checkpoint_without_routes_keeps_ordinary_values():
    state = state_for(torch.tensor([1], dtype=torch.int16), tokens=torch.arange(8))
    del state["field_data"]["routed_experts"]
    payload = packed(state)
    assert b"_rebuild_device_tensor_from_numpy" not in payload
    assert torch.equal(pickle.loads(payload)["field_data"]["tokens"][0], torch.arange(8))


@pytest.mark.parametrize("legacy", [False, True])
def test_real_save_load_handler_and_unmodified_reader(tmp_path, legacy):
    route = torch.arange(160, dtype=torch.int16).reshape(4, 40, 1) % 16
    path, actor = tmp_path / "checkpoint.pkl", unit()
    if legacy:
        path.write_bytes(pickle.dumps(state_for(route), protocol=pickle.HIGHEST_PROTOCOL))
    else:
        actor.storage_data.put_data({"routed_experts": [route]}, [0])
        reply = actor._handle_save_checkpoint(request(path))
        assert reply.body["success"] and reply.body["failure"] is None
    loaded = unit()
    assert loaded._handle_load_checkpoint(request(path)).body["success"]
    restored = loaded.storage_data.get_data(["routed_experts"], [0])["routed_experts"][0]
    assert torch.equal(restored, route) and restored.dtype == torch.int16
    # Isolated Python imports no candidate SDK; the installed Torch/pickle reader
    # must reconstruct the exact original tensor without a new decoding branch.
    code = (
        "import pickle,sys,torch; "
        "v=pickle.load(open(sys.argv[1],'rb'))['field_data']['routed_experts'][0]; "
        "assert v.dtype==torch.int16 and tuple(v.shape)==(4,40,1); "
        "assert torch.equal(v,torch.arange(160,dtype=torch.int16).reshape(4,40,1)%16)"
    )
    packed_path = tmp_path / "ordinary-reader.pkl"
    packed_path.write_bytes(packed(state_for(route)))
    subprocess.run([sys.executable, "-I", "-c", code, str(packed_path)], check=True, timeout=30)


def test_async_capture_survives_clear_and_reuse(tmp_path, monkeypatch):
    actor, entered, release = unit(), Event(), Event()
    original = _CheckpointPickler.dump

    def blocked(self, state):
        entered.set()
        assert release.wait(10)
        return original(self, state)

    monkeypatch.setattr(_CheckpointPickler, "dump", blocked)
    old = torch.tensor([1, 15], dtype=torch.int16)
    actor.storage_data.put_data({"routed_experts": [old]}, [0])
    path = tmp_path / "async.pkl"
    try:
        reply = actor._handle_save_checkpoint(request(path, phase="prepare", indexes=[0], max_snapshot_bytes=1 << 20))
        assert reply.body["prepared"] and entered.wait(10)
        actor.storage_data.clear([0])
        actor.storage_data.put_data({"routed_experts": [torch.tensor([9], dtype=torch.int16)]}, [0])
    finally:
        release.set()
    actor._checkpoint_future.result(timeout=10)
    restored = pickle.loads(path.read_bytes())
    assert torch.equal(restored["field_data"]["routed_experts"][0], old)
    assert restored["active_keys"] == {0}


@pytest.mark.parametrize("failure", ["dump", "fsync"])
def test_real_writer_failures_are_retained(tmp_path, monkeypatch, failure):
    actor = unit()
    actor.storage_data.put_data({"routed_experts": [torch.tensor([1], dtype=torch.int16)]}, [0])

    def fail(*args):
        raise OSError(5, "checkpoint fixture write failed")

    if failure == "dump":
        monkeypatch.setattr(_CheckpointPickler, "dump", fail)
    else:
        monkeypatch.setattr(os, "fsync", fail)
    path = tmp_path / "failed.pkl"
    reply = actor._handle_save_checkpoint(request(path, phase="prepare", indexes=[0], max_snapshot_bytes=1 << 20))
    assert reply.body["prepared"]
    with pytest.raises(OSError, match="checkpoint fixture write failed"):
        actor._checkpoint_future.result(timeout=10)
    status = actor._handle_save_checkpoint(request(path, phase="status")).body
    assert status["done"] and status["failure"]["storage_io"] and status["failure"]["errno"] == 5


def test_matched_serialization_and_fsync_cost(tmp_path):
    routes = {index: torch.randint(0, 16, (8192, 40, 1), dtype=torch.int16) for index in range(128)}
    state = state_for(routes[0], tokens=torch.arange(8192))
    state["field_data"]["routed_experts"], state["active_keys"] = routes, set(routes)
    started = time.perf_counter()
    assert all(value.numpy().min() >= 0 and value.numpy().max() <= 255 for value in routes.values())
    range_s = time.perf_counter() - started
    started = time.perf_counter()
    copies = [value.numpy().astype(np.uint8) for value in routes.values()]
    narrow_copy_s = time.perf_counter() - started
    del copies
    measurements = {
        "input_route_bytes": sum(value.nbytes for value in routes.values()),
        "range_check_s": range_s,
        "narrow_copy_s": narrow_copy_s,
    }
    for mode in ("legacy", "packed"):
        started = time.perf_counter()
        payload = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL) if mode == "legacy" else packed(state)
        measurements[f"{mode}_serialize_s"] = time.perf_counter() - started
        measurements[f"{mode}_bytes"] = len(payload)
        for label, root in (("local", str(tmp_path)), ("shared", os.environ.get("CHECKPOINT_BENCH_WRITE_ROOT"))):
            if root is None:
                continue
            with tempfile.TemporaryDirectory(prefix="route-packing-", dir=root) as directory:
                path = os.path.join(directory, "payload.pkl")
                started = time.perf_counter()
                with open(path, "wb") as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
                measurements[f"{mode}_{label}_write_flush_fsync_s"] = time.perf_counter() - started
                assert os.path.getsize(path) == len(payload)
        restored = pickle.loads(payload)
        for index, value in routes.items():
            actual = restored["field_data"]["routed_experts"][index]
            assert actual.dtype == value.dtype and actual.shape == value.shape and torch.equal(actual, value)
        assert restored["active_keys"] == state["active_keys"]
        del restored, payload
    assert measurements["packed_bytes"] < measurements["legacy_bytes"] * 0.6
    print("CHECKPOINT_ROUTE_BENCH " + json.dumps(measurements, sort_keys=True))


@pytest.mark.parametrize("layout", ["rows", "nested", "dense"])
def test_public_transport_put_save_route_eligibility(tmp_path, layout):
    rows = [torch.randint(0, 16, (32768 + index * (layout != "dense"), 40, 1), dtype=torch.int16) for index in range(4)]
    carrier = rows
    if layout == "nested":
        carrier = torch.nested.as_nested_tensor(rows, layout=torch.jagged)
    elif layout == "dense":
        carrier = torch.stack(rows)
    message = ZMQMessage.create(
        ZMQRequestType.PUT_DATA, "test", {"global_indexes": list(range(4)), "data": {"routed_experts": carrier}}
    )
    started = time.perf_counter()
    received = ZMQMessage.deserialize(message.serialize())
    costs = {"wire_encode_decode_s": time.perf_counter() - started}
    actor = unit()
    started = time.perf_counter()
    assert actor._handle_put(received).request_type == ZMQRequestType.PUT_DATA_RESPONSE
    costs["put_s"] = time.perf_counter() - started
    values = dict(enumerate(actor.storage_data.get_data(["routed_experts"], list(range(4)))["routed_experts"]))
    state = state_for(values[0])
    state["field_data"]["routed_experts"], state["active_keys"] = values, set(values)
    started = time.perf_counter()
    legacy = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
    costs["legacy_serialize_s"] = time.perf_counter() - started
    started = time.perf_counter()
    assert all(value.numpy().min() >= 0 and value.numpy().max() <= 255 for value in values.values())
    costs["range_check_s"] = time.perf_counter() - started
    started = time.perf_counter()
    narrow = [value.numpy().astype(np.uint8) for value in values.values()]
    costs["narrow_copy_s"] = time.perf_counter() - started
    del narrow
    started = time.perf_counter()
    direct = packed(state)
    costs["packed_serialize_s"] = time.perf_counter() - started
    path = tmp_path / "transport.pkl"
    started = time.perf_counter()
    assert actor._handle_save_checkpoint(request(path)).body["failure"] is None
    costs["packed_handler_serialize_write_fsync_s"] = time.perf_counter() - started
    payload = path.read_bytes()
    assert b"_rebuild_device_tensor_from_numpy" in direct
    assert len(payload) < len(legacy) * 0.6
    restored = pickle.loads(path.read_bytes())["field_data"]["routed_experts"]
    ordinary = pickle.loads(legacy)["field_data"]["routed_experts"]
    for index, expected in enumerate(rows):
        assert restored[index].dtype == expected.dtype and restored[index].shape == expected.shape
        assert torch.equal(restored[index], expected)
    for index in restored:
        for other in restored:
            assert (restored[index] is restored[other]) == (ordinary[index] is ordinary[other])
            assert (restored[index].untyped_storage().data_ptr() == restored[other].untyped_storage().data_ptr()) == (
                ordinary[index].untyped_storage().data_ptr() == ordinary[other].untyped_storage().data_ptr()
            )
    restored[0].fill_(99)
    ordinary[0].fill_(99)
    assert all(torch.equal(restored[index], ordinary[index]) for index in restored)
    loaded = unit()
    assert loaded._handle_load_checkpoint(request(path)).body["success"]
    for index, expected in enumerate(rows):
        assert torch.equal(loaded.storage_data.get_data(["routed_experts"], [index])["routed_experts"][0], expected)
    legacy_path = tmp_path / "legacy.pkl"
    legacy_path.write_bytes(legacy)
    code = (
        "import pickle,sys,torch; a=pickle.load(open(sys.argv[1],'rb')); b=pickle.load(open(sys.argv[2],'rb')); "
        "assert a['active_keys']==b['active_keys']; "
        "a=a['field_data']['routed_experts']; b=b['field_data']['routed_experts']; "
        "assert a.keys()==b.keys(); "
        "assert all(a[k].dtype==b[k].dtype and a[k].shape==b[k].shape and torch.equal(a[k],b[k]) for k in a)"
    )
    packed_path = tmp_path / "ordinary-reader.pkl"
    packed_path.write_bytes(direct)
    subprocess.run([sys.executable, "-I", "-c", code, str(packed_path), str(legacy_path)], check=True, timeout=30)
    for mode, data in (("legacy", legacy), ("packed", direct)):
        root = os.environ.get("CHECKPOINT_BENCH_WRITE_ROOT", str(tmp_path))
        with tempfile.TemporaryDirectory(prefix="route-wire-", dir=root) as directory:
            started = time.perf_counter()
            with open(os.path.join(directory, "payload.pkl"), "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            costs[mode + "_write_flush_fsync_s"] = time.perf_counter() - started
    print(
        "TRANSPORT_ROUTE_PACKING "
        + json.dumps(
            {
                "layout": layout,
                "legacy_bytes": len(legacy),
                "saved_bytes": len(payload),
                "packed": b"_rebuild_device_tensor_from_numpy" in direct,
                "costs": costs,
                "rows": [
                    {
                        "view": value._base is not None,
                        "offset": value.storage_offset(),
                        "logical_bytes": value.nbytes,
                        "backing_bytes": value.untyped_storage().nbytes(),
                    }
                    for value in values.values()
                ],
            }
        )
    )
