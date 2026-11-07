# Copyright 2026 Zyphra Technologies Inc.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Cleared rows return their allocator pages instead of pinning the high-water RSS."""

import psutil
import pytest
import torch
from tensordict import TensorDict

from transfer_queue.storage import simple_storage
from transfer_queue.storage.managers.simple_storage_manager import AsyncSimpleStorageManager
from transfer_queue.storage.simple_storage import SimpleStorageUnit, StorageUnitData
from transfer_queue.utils.zmq_utils import ZMQMessage, ZMQRequestType

pytestmark = pytest.mark.skipif(simple_storage._libc_malloc_trim() is None, reason="requires glibc malloc_trim")

ROWS, TOKENS = 4096, 12_000  # 96 KB row frames: below glibc's mmap threshold, so arena-allocated


def unit():
    actor = object.__new__(SimpleStorageUnit.__ray_metadata__.modified_class)
    actor.storage_unit_id, actor.storage_unit_size = "release-test", None
    actor.storage_data = StorageUnitData(None)
    return actor


def put_rows(actor, keys):
    rows = [torch.randint(0, 1000, (TOKENS,)) for _ in keys]
    data = TensorDict({"input_ids": torch.nested.as_nested_tensor(rows, layout=torch.jagged)}, batch_size=[len(keys)])
    selected = {"input_ids": AsyncSimpleStorageManager._select_by_positions(data["input_ids"], list(range(len(keys))))}
    request = ZMQMessage.create(ZMQRequestType.PUT_DATA, "test", {"global_indexes": keys, "data": selected})
    frames = [bytes(memoryview(frame)) for frame in request.serialize()]
    reply = actor._handle_put(ZMQMessage.deserialize(frames, retain_packed_routes=True))
    assert reply.request_type == ZMQRequestType.PUT_DATA_RESPONSE


def clear(actor, keys):
    reply = actor._handle_clear(ZMQMessage.create(ZMQRequestType.CLEAR_DATA, "test", {"global_indexes": keys}))
    assert reply.request_type == ZMQRequestType.CLEAR_DATA_RESPONSE
    trimmer = getattr(actor, "_trimmer", None)
    if trimmer is not None:
        trimmer.join(timeout=60)


def test_clearing_most_rows_returns_their_pages(monkeypatch):
    actor, process = unit(), psutil.Process()
    for start in range(0, ROWS, 256):
        put_rows(actor, list(range(start, start + 256)))
    # Keep one row in eight so freed frames sit between live ones, as rows drain in production.
    drained = [key for key in range(ROWS) if key % 8]
    monkeypatch.setattr(simple_storage, "_TRIM_MIN_RELEASED_BYTES", 1 << 62)
    held = process.memory_info().rss
    clear(actor, drained[: len(drained) // 2])
    untrimmed = process.memory_info().rss
    monkeypatch.setattr(simple_storage, "_TRIM_MIN_RELEASED_BYTES", 0)
    clear(actor, drained[len(drained) // 2 :])
    trimmed = process.memory_info().rss
    released = len(drained) * TOKENS * 8
    assert held - untrimmed < released // 4, "without a trim the first half stays resident"
    assert held - trimmed > released // 2
    survivors = actor.storage_data.get_data(["input_ids"], [0, 8])["input_ids"]
    assert [row.numel() for row in survivors] == [TOKENS, TOKENS]


def test_small_clears_accumulate_until_a_tenth_of_the_payload(monkeypatch):
    calls = []
    monkeypatch.setattr(simple_storage, "_libc_malloc_trim", lambda: lambda pad: calls.append(pad) or 1)
    monkeypatch.setattr(simple_storage, "_TRIM_MIN_RELEASED_BYTES", 0)
    actor = unit()
    put_rows(actor, list(range(100)))
    clear(actor, [0, 1, 2, 3, 4])
    assert calls == []
    clear(actor, [5, 6, 7, 8, 9])
    assert calls == [0]
    assert actor._released_since_trim == 0
