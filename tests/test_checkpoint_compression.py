# Copyright 2026 Zyphra Technologies Inc.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Lossless compressed shards keep the existing storage cut and failure contract."""

import hashlib
import io
import json
import pickle
import struct
import time
from threading import Event

import numpy as np
import pytest
import torch
import zstandard

from transfer_queue.storage.simple_storage import (
    SimpleStorageUnit,
    StorageUnitData,
    _CheckpointHashStream,
    _CheckpointPickler,
    _read_storage_checkpoint,
    _write_storage_checkpoint,
)
from transfer_queue.utils.zmq_utils import ZMQMessage, ZMQRequestType


def unit():
    actor = object.__new__(SimpleStorageUnit.__ray_metadata__.modified_class)
    actor.storage_unit_id, actor.storage_unit_size = "compressed-test", 8
    actor.storage_data = StorageUnitData(8)
    return actor


def request(path, **body):
    return ZMQMessage.create(ZMQRequestType.SAVE_STORAGE_CHECKPOINT, "test", {"path": str(path), **body})


def state():
    backing = torch.arange(24, dtype=torch.int64)
    return {
        "storage_unit_id": "compressed-test",
        "storage_unit_size": 8,
        "field_data": {
            "input_ids": {0: backing, 1: backing[2:8]},
            "position_ids": {0: backing},
            "response_mask": {0: torch.tensor([False, True, True])},
            "routed_experts": {0: torch.arange(160, dtype=torch.int16).reshape(4, 40, 1) % 16},
            "log_probs": {0: torch.tensor([-0.0, float("inf"), float("nan")], dtype=torch.float32)},
            "metadata": {0: {"route": "train-route", "text": "\u03bb\n", "raw": bytes(range(256))}},
            "numpy": {0: np.arange(12, dtype=np.int32).reshape(3, 4)},
        },
        "active_keys": {0, 1},
    }


@pytest.mark.parametrize("legacy", [False, True])
def test_exact_tensor_bytes_dtypes_shapes_objects_and_aliases(tmp_path, legacy):
    original = state()
    path = tmp_path / "checkpoint.pkl"
    if legacy:
        path.write_bytes(pickle.dumps(original, protocol=pickle.HIGHEST_PROTOCOL))
    else:
        _write_storage_checkpoint(path, original)
    actual = _read_storage_checkpoint(path)
    baseline = pickle.loads(pickle.dumps(original, protocol=pickle.HIGHEST_PROTOCOL))
    fields = actual["field_data"]
    for name, values in original["field_data"].items():
        for index, value in values.items():
            restored = fields[name][index]
            if isinstance(value, torch.Tensor):
                assert restored.dtype == value.dtype and restored.shape == value.shape
                assert torch.equal(restored.contiguous().view(torch.uint8), value.contiguous().view(torch.uint8))
            elif isinstance(value, np.ndarray):
                assert restored.dtype == value.dtype and np.array_equal(restored, value)
            else:
                assert restored == value
    assert fields["input_ids"][0] is fields["position_ids"][0]
    fields["input_ids"][1][0] = 91
    baseline["field_data"]["input_ids"][1][0] = 91
    assert torch.equal(fields["input_ids"][0], baseline["field_data"]["input_ids"][0])
    assert actual["active_keys"] == original["active_keys"]


@pytest.mark.parametrize(
    "change", ["truncate", "flip", "append", "header", "checksumless", "trailing_pickle", "hash", "truncate_relength"]
)
def test_corruption_refuses_before_replacing_live_storage(tmp_path, change):
    path = tmp_path / "broken.pkl"
    _write_storage_checkpoint(path, state())
    payload = bytearray(path.read_bytes())
    if change == "truncate":
        del payload[-5:]
    elif change == "truncate_relength":
        del payload[-2:]
        payload[8:16] = struct.pack("<Q", len(payload))
    elif change == "flip":
        payload[-1] ^= 1
    elif change == "append":
        payload.extend(b"uncommitted")
    elif change == "header":
        payload[8] ^= 1
    elif change == "hash":
        payload[16] ^= 1
    else:
        plain = pickle.dumps(state()) + (b"unexpected" if change == "trailing_pickle" else b"")
        frame = zstandard.ZstdCompressor(write_checksum=change != "checksumless").compress(plain)
        payload = payload[:8] + struct.pack("<Q", 48 + len(frame)) + hashlib.sha256(frame).digest() + frame
    path.write_bytes(payload)
    actor = unit()
    old = torch.tensor([77])
    actor.storage_data.put_data({"input_ids": [old]}, [7])
    response = actor._handle_load_checkpoint(request(path)).body
    assert not response["success"]
    assert actor.storage_data.field_data == {"input_ids": {7: old}}
    assert actor.storage_data._active_keys == {7}


def test_async_snapshot_survives_live_clear_and_key_reuse(tmp_path, monkeypatch):
    actor, entered, release = unit(), Event(), Event()
    original = _CheckpointPickler.dump

    def blocked(self, value):
        entered.set()
        assert release.wait(10)
        return original(self, value)

    monkeypatch.setattr(_CheckpointPickler, "dump", blocked)
    actor.storage_data.put_data({"routed_experts": [torch.tensor([0, 15], dtype=torch.int16)]}, [0])
    path = tmp_path / "snapshot.pkl"
    try:
        accepted = actor._handle_save_checkpoint(
            request(path, phase="prepare", indexes=[0], max_snapshot_bytes=1 << 20)
        ).body
        assert accepted["prepared"] and entered.wait(10)
        actor.storage_data.clear([0])
        actor.storage_data.put_data({"routed_experts": [torch.tensor([99], dtype=torch.int16)]}, [0])
    finally:
        release.set()
    actor._checkpoint_future.result(timeout=10)
    assert torch.equal(_read_storage_checkpoint(path)["field_data"]["routed_experts"][0], torch.tensor([0, 15]))
    assert torch.equal(actor.storage_data.field_data["routed_experts"][0], torch.tensor([99]))


def test_unknown_ownership_still_uses_synchronous_path(tmp_path):
    actor = unit()
    actor.storage_data.put_data({"opaque": [range(3)]}, [0])
    path = tmp_path / "fallback.pkl"
    reply = actor._handle_save_checkpoint(request(path, phase="prepare", indexes=[0], max_snapshot_bytes=1)).body
    assert not reply["prepared"] and reply["done"] and reply["failure"] is None
    assert _read_storage_checkpoint(path)["field_data"]["opaque"][0] == range(3)


def test_compression_reduces_repeated_fields_without_changing_values(tmp_path):
    original = state()
    original["field_data"]["position_ids"] = {0: torch.arange(65536, dtype=torch.int64).repeat(3, 1)}
    path = tmp_path / "compressed.pkl"
    _write_storage_checkpoint(path, original)
    plain = pickle.dumps(original, protocol=pickle.HIGHEST_PROTOCOL)
    assert path.stat().st_size < len(plain) / 2
    assert torch.equal(
        _read_storage_checkpoint(path)["field_data"]["position_ids"][0], original["field_data"]["position_ids"][0]
    )


@pytest.mark.parametrize("phase", ["pickle", "compressed_write", "fsync"])
def test_async_writer_abort_keeps_original_error_and_no_success(tmp_path, monkeypatch, phase):
    actor = unit()
    actor.storage_data.put_data({"routed_experts": [torch.tensor([0, 15], dtype=torch.int16)]}, [0])

    def fail(*args, **kwargs):
        raise OSError(5, "compressed checkpoint fixture failure")

    if phase == "pickle":
        monkeypatch.setattr(_CheckpointPickler, "dump", fail)
    elif phase == "compressed_write":
        monkeypatch.setattr(_CheckpointHashStream, "write", fail)
    else:
        monkeypatch.setattr("transfer_queue.storage.simple_storage.os.fsync", fail)
    path = tmp_path / "failed.pkl"
    accepted = actor._handle_save_checkpoint(request(path, phase="prepare", indexes=[0], max_snapshot_bytes=1 << 20))
    assert accepted.body["prepared"]
    with pytest.raises(OSError, match="compressed checkpoint fixture failure"):
        actor._checkpoint_future.result(timeout=10)
    failure = actor._handle_save_checkpoint(request(path, phase="status")).body
    assert failure["done"] and failure["failure"]["storage_io"] and failure["failure"]["errno"] == 5


def test_paired_pickle_compression_fsync_and_restore_cost(tmp_path):
    original = state()
    original["field_data"].pop("numpy")
    original["field_data"]["routed_experts"] = {
        index: torch.randint(0, 16, (32768, 40, 1), dtype=torch.int16) for index in range(16)
    }
    original["field_data"]["input_ids"] = {index: torch.arange(32768) for index in range(16)}
    original["field_data"]["position_ids"] = {index: torch.arange(32768) for index in range(16)}
    original["active_keys"] = set(range(16))
    stream = io.BytesIO()
    started = time.perf_counter()
    _CheckpointPickler(stream, original).dump(original)
    packed_s = time.perf_counter() - started
    path = tmp_path / "compressed.pkl"
    started = time.perf_counter()
    _write_storage_checkpoint(path, original)
    write_s = time.perf_counter() - started
    started = time.perf_counter()
    restored = _read_storage_checkpoint(path)
    read_s = time.perf_counter() - started
    for name in ("routed_experts", "input_ids", "position_ids"):
        for index, value in original["field_data"][name].items():
            actual = restored["field_data"][name][index]
            assert actual.dtype == value.dtype and actual.shape == value.shape and torch.equal(actual, value)
    assert path.stat().st_size < stream.tell()
    print(
        "CHECKPOINT_ZSTD_BENCH "
        + json.dumps(
            {
                "population": "Synthetic16rows; no productiondensity/timingextrapolation",
                "route_logical_bytes": sum(value.nbytes for value in original["field_data"]["routed_experts"].values()),
                "packed_pickle_bytes": stream.tell(),
                "zstd_bytes": path.stat().st_size,
                "packed_pickle_s": packed_s,
                "zstd_write_flush_fsync_s": write_s,
                "zstd_restore_s": read_s,
            },
            sort_keys=True,
        )
    )
