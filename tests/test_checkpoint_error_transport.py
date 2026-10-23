# Copyright 2026 Zyphra Technologies Inc.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Checkpoint failures cross Ray without transferring local writer ownership."""

import pickle
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

import pytest
import ray
import ray.cloudpickle

import transfer_queue.interface as interface
from transfer_queue.storage.managers.simple_storage_manager import AsyncSimpleStorageManager


@pytest.mark.parametrize("serializer", [pickle, ray.cloudpickle], ids=["pickle", "ray-cloudpickle"])
@pytest.mark.parametrize("timed_out", [False, True], ids=["write-error", "deadline"])
def test_error_copy_omits_future_without_changing_local_ownership(serializer, timed_out):
    kind = interface.CheckpointStorageTimeoutError if timed_out else interface.CheckpointStorageError
    failures = () if timed_out else ({"storage_unit_id": "unit-2", "storage_io": True, "type": "OSError"},)
    error = kind("original storage failure", failures=failures, timed_out=timed_out)
    future = Future()
    error.checkpoint_future = future
    future.set_exception(error)

    copied = serializer.loads(serializer.dumps(error))

    assert type(copied) is kind
    assert copied.args == error.args
    assert copied.failures == failures
    assert copied.timed_out is timed_out
    assert copied.checkpoint_future is None
    assert error.checkpoint_future is future
    assert future.exception() is error


@pytest.fixture(scope="module")
def private_ray(tmp_path_factory):
    assert not ray.is_initialized()
    ray.init(
        num_cpus=1,
        num_gpus=0,
        include_dashboard=False,
        object_store_memory=128 * 1024 * 1024,
        _memory=1024 * 1024 * 1024,
        _node_ip_address="127.0.0.1",
        _temp_dir=str(tmp_path_factory.mktemp("error-ray")),
    )
    try:
        yield
    finally:
        ray.shutdown()


@ray.remote(num_cpus=1)
class CheckpointOwner:
    def fail(self, checkpoint_dir, timed_out):
        self.path = Path(checkpoint_dir)
        self.settled = False
        self.failures = (
            []
            if timed_out
            else [{"storage_unit_id": "unit-2", "storage_io": True, "type": "OSError", "message": "write failed"}]
        )

        def controller(path, **kwargs):
            Path(path).write_bytes(b"native-controller-cut")
            return {"indexes": []}

        def storage(path, *, phase, **kwargs):
            if phase == "prepare":
                return {"prepared": True, "retained_bytes": 0}
            assert phase == "status"
            return {"done": self.settled, "failures": self.failures}

        manager = object.__new__(AsyncSimpleStorageManager)
        manager.controller_handshake_socket = None
        manager._owns_zmq_context = False
        manager.zmq_context = None
        interface._TQ_CONTROLLER = object()
        interface._TQ_CLIENT = SimpleNamespace(
            storage_manager=manager, save_controller_checkpoint=controller, save_storage_checkpoint=storage
        )
        self.future = interface.save_checkpoint(
            self.path, async_storage=True, max_snapshot_bytes=1024, timeout_s=0.05 if timed_out else 5
        )
        self.future.result(timeout=10)

    def settle_and_retire(self):
        import shutil

        error = self.future.exception()
        assert error.checkpoint_future is self.future
        assert interface.checkpoint_status(self.path, self.future) == {"settled": False}
        with pytest.raises(RuntimeError, match="unsettled"):
            interface.retire_checkpoint(self.path, self.future)
        with pytest.raises(ValueError, match="owned pending"):
            interface.checkpoint_status(self.path, Future())
        self.settled = True
        assert interface.checkpoint_status(self.path, self.future) == {"settled": True}
        staging = self.path.parent / f"{self.path.name}.tmp"
        assert staging.is_dir() and not self.path.exists()
        shutil.rmtree(staging)
        interface.retire_checkpoint(self.path, self.future)
        assert error.checkpoint_future is self.future
        assert self.future.exception() is error
        assert not self.path.exists() and not staging.exists()
        return True


@pytest.mark.parametrize("timed_out", [False, True], ids=["write-error", "deadline"])
def test_public_save_error_crosses_real_ray_and_keeps_local_retirement(private_ray, tmp_path, timed_out):
    owner = CheckpointOwner.remote()
    kind = interface.CheckpointStorageTimeoutError if timed_out else interface.CheckpointStorageError
    try:
        with pytest.raises(ray.exceptions.RayTaskError) as raised:
            ray.get(owner.fail.remote(str(tmp_path / "failed-cut"), timed_out), timeout=30)
        error = raised.value.cause
        assert type(error) is kind
        assert error.timed_out is timed_out
        assert error.failures == (
            ()
            if timed_out
            else ({"storage_unit_id": "unit-2", "storage_io": True, "type": "OSError", "message": "write failed"},)
        )
        assert error.checkpoint_future is None
        assert ray.get(owner.settle_and_retire.remote(), timeout=30)
    finally:
        ray.kill(owner, no_restart=True)
