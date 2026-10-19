# Copyright 2026 Zyphra Technologies Inc.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Save receipts describe existing replies without deciding checkpoint state."""

import asyncio
import json
from concurrent.futures import Future
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import transfer_queue.interface as interface
from transfer_queue.storage.managers.simple_storage_manager import AsyncSimpleStorageManager


def manager_with_replies(*, fallback=False):
    manager = object.__new__(AsyncSimpleStorageManager)
    manager.storage_unit_infos = {f"unit-{index}": SimpleNamespace(ip=f"192.0.2.{index + 1}") for index in range(8)}
    replies = {
        key: {"prepared": not (fallback and index == 7), "retained_bytes": None if index == 7 else index + 10}
        for index, key in enumerate(manager.storage_unit_infos)
    }

    async def save(path, target_storage_unit, **kwargs):
        if kwargs["phase"] == "prepare":
            return {**replies[target_storage_unit], "opaque": object()}
        return {"done": True, "failure": None}

    manager._save_single_storage_unit = AsyncMock(side_effect=save)
    return manager, replies


@pytest.mark.parametrize("fallback", [False, True])
def test_eight_prepare_replies_retain_exact_identity_bytes_and_fallback(tmp_path, fallback):
    manager, replies = manager_with_replies(fallback=fallback)
    result = asyncio.run(manager.save_checkpoint(str(tmp_path), phase="prepare", indexes=[3], max_snapshot_bytes=800))
    assert result["prepared"] is not fallback
    assert result["retained_bytes"] == sum(item["retained_bytes"] or 0 for item in replies.values())
    assert result["storage_units"] == [
        {"storage_unit_id": key, "ip": f"192.0.2.{index + 1}", **item, "max_snapshot_bytes": 100}
        for index, (key, item) in enumerate(replies.items())
    ]
    assert manager._save_single_storage_unit.await_count == 8
    assert all(call.kwargs["indexes"] == [3] for call in manager._save_single_storage_unit.await_args_list)


@pytest.mark.parametrize("invalid", ["ip", "boolean_size", "negative_size"])
def test_invalid_optional_receipt_does_not_change_preparation(tmp_path, invalid):
    manager, replies = manager_with_replies()
    if invalid == "ip":
        manager.storage_unit_infos["unit-0"].ip = "not-an-IP-or-a-receipt-field"
    else:
        replies["unit-0"]["retained_bytes"] = True if invalid == "boolean_size" else -1
    result = asyncio.run(manager.save_checkpoint(str(tmp_path), phase="prepare", indexes=[], max_snapshot_bytes=800))
    assert result["prepared"] is True
    assert "storage_units" not in result
    assert manager._save_single_storage_unit.await_count == 8


def test_prepare_error_is_not_optional_diagnostic_failure(tmp_path):
    manager, _ = manager_with_replies()
    error = RuntimeError("original preparation transport error")

    async def save(path, target_storage_unit, **kwargs):
        if target_storage_unit == "unit-3":
            raise error
        return {"prepared": True, "retained_bytes": 10}

    manager._save_single_storage_unit.side_effect = save
    with pytest.raises(RuntimeError) as raised:
        asyncio.run(manager.save_checkpoint(str(tmp_path), phase="prepare", indexes=[], max_snapshot_bytes=800))
    assert raised.value is error
    assert manager._save_single_storage_unit.await_count == 8


def install_client(monkeypatch, *, fallback=False, invalid_receipt=False, gate=None):
    manager, _ = manager_with_replies(fallback=fallback)
    if invalid_receipt:
        manager.storage_unit_infos["unit-0"].ip = "invalid"

    def controller(path, **kwargs):
        Path(path).write_bytes(b"existing-controller-cut")
        return {"indexes": [3]}

    def storage(path, **kwargs):
        if kwargs["phase"] == "status" and gate is not None:
            assert gate.wait(5), "owned test writer was not released"
        return asyncio.run(manager.save_checkpoint(path, **kwargs))

    monkeypatch.setattr(interface, "_TQ_CONTROLLER", object())
    monkeypatch.setattr(
        interface,
        "_TQ_CLIENT",
        SimpleNamespace(
            storage_manager=manager, save_controller_checkpoint=controller, save_storage_checkpoint=storage
        ),
    )
    monkeypatch.setattr(interface, "_PENDING_CHECKPOINT", None)
    monkeypatch.setattr(interface, "_PENDING_CHECKPOINT_DIR", None)


@pytest.mark.parametrize("fallback", [False, True])
def test_committed_receipt_preserves_future_contract_and_durable_order(tmp_path, monkeypatch, fallback):
    gate = Event()
    if fallback:
        gate.set()
    install_client(monkeypatch, fallback=fallback, gate=gate)
    path = tmp_path / "cut"
    future = interface.save_checkpoint(path, async_storage=True, max_snapshot_bytes=800, metadata={"global_steps": 2})
    try:
        if fallback:
            assert future is None
        else:
            assert isinstance(future, Future)
            assert not future.done()
            assert not path.exists()
    finally:
        gate.set()
        if future is not None:
            assert future.result(timeout=5) is None
    metadata = json.loads((path / "metadata.json").read_text())
    receipt = metadata["storage_checkpoint"]
    assert metadata["user_metadata"] == {"global_steps": 2}
    assert receipt["mode"] == ("synchronous_fallback" if fallback else "async")
    assert receipt["max_snapshot_bytes"] == 800
    assert len(receipt["storage_units"]) == 8
    assert receipt["storage_units"][-1]["retained_bytes"] is None
    assert receipt["storage_units"][-1]["prepared"] is not fallback
    assert (path / "controller_state.pkl").read_bytes() == b"existing-controller-cut"


@pytest.mark.parametrize("failure", ["invalid_primitives", "construction"])
def test_optional_receipt_failure_still_commits(tmp_path, monkeypatch, failure):
    install_client(monkeypatch, invalid_receipt=failure == "invalid_primitives")
    if failure == "construction":
        original_dumps = interface.json.dumps

        def fail_receipt(value, *args, **kwargs):
            if isinstance(value, dict) and "storage_units" in value:
                raise ValueError("receipt")
            return original_dumps(value, *args, **kwargs)

        monkeypatch.setattr(interface.json, "dumps", fail_receipt)
    path = tmp_path / "cut"
    future = interface.save_checkpoint(path, async_storage=True, max_snapshot_bytes=800)
    assert future.result(timeout=5) is None
    metadata = json.loads((path / "metadata.json").read_text())
    assert "storage_checkpoint" not in metadata
    assert metadata["storage_saved"] is True


def test_real_metadata_io_failure_remains_failure_and_preserves_prior_cut(tmp_path, monkeypatch):
    install_client(monkeypatch)
    old = tmp_path / "prior" / "metadata.json"
    old.parent.mkdir()
    old.write_text('{"prior": true}')
    error = OSError("original metadata write failure")
    original_dump = interface.json.dump

    def fail_metadata(value, stream, **kwargs):
        if Path(stream.name).name == "metadata.json":
            raise error
        return original_dump(value, stream, **kwargs)

    monkeypatch.setattr(interface.json, "dump", fail_metadata)
    path = tmp_path / "new"
    future = interface.save_checkpoint(path, async_storage=True, max_snapshot_bytes=800)
    with pytest.raises(OSError) as raised:
        future.result(timeout=5)
    assert raised.value is error
    assert not path.exists()
    assert json.loads(old.read_text()) == {"prior": True}
