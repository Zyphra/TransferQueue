# Copyright 2025 The TransferQueue Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Executor sizing must inspect Ray without starting it for non-Ray clients."""

import multiprocessing
import os

import numpy as np
import pytest
import ray
import torch
from tensordict import TensorDict

from transfer_queue.controller import TransferQueueController
from transfer_queue.metadata import BatchMeta, extract_field_schema
from transfer_queue.storage.clients.base import StorageClientFactory, StorageKVClient
from transfer_queue.storage.managers.base import (
    LIMIT_THREADS_PER_MANAGER_IN_DRIVER,
    LIMIT_THREADS_PER_MANAGER_IN_RAY_ACTOR,
    KVStorageManager,
)


@StorageClientFactory.register("ExecutorTestMemoryClient")
class MemoryClient(StorageKVClient):
    """An ordinary registered KV backend for the non-Ray driver's real manager."""

    def __init__(self, config):
        super().__init__(config)
        self.values = {}

    def put(self, keys, values):
        self.values.update(zip(keys, values, strict=True))

    def get(self, keys, shapes=None, dtypes=None, custom_backend_meta=None):
        return [self.values[key] for key in keys]

    def clear(self, keys, custom_backend_meta=None):
        for key in keys:
            self.values.pop(key, None)


def controller_process(connection):
    """Keep the controller's Ray lifetime separate from the non-Ray driver."""
    ray.init(address="local", num_cpus=1, num_gpus=0, object_store_memory=256 * 1024 * 1024)
    controller = TransferQueueController.remote()
    try:
        connection.send(ray.get(controller.get_zmq_server_info.remote()))
        connection.recv()
    finally:
        ray.kill(controller)
        ray.shutdown()
        connection.close()


@pytest.fixture
def external_controller():
    assert not ray.is_initialized()
    context = multiprocessing.get_context("spawn")
    connection, child_connection = context.Pipe()
    process = context.Process(target=controller_process, args=(child_connection,))
    process.start()
    child_connection.close()
    try:
        assert connection.poll(90), "Real controller did not become ready"
        yield connection.recv()
    finally:
        if process.is_alive():
            connection.send(None)
        process.join(timeout=60)
        if process.is_alive():
            process.terminate()
            process.join(timeout=10)
        connection.close()
    assert process.exitcode == 0


def merge(manager):
    data = TensorDict(
        {
            "tokens": torch.nested.as_nested_tensor([torch.tensor([1, 2]), torch.tensor([3])], layout=torch.jagged),
            "scores": torch.tensor([0.5, 1.0]),
        },
        batch_size=2,
    )
    meta = BatchMeta(
        global_indexes=[0, 1],
        partition_ids=["merge"] * 2,
        field_schema=extract_field_schema(data),
        production_status=np.ones(2, dtype=np.int8),
    )
    result = manager._merge_tensors_to_tensordict(meta, manager._generate_values(data))
    torch.testing.assert_close(result["scores"], data["scores"])
    for actual, expected in zip(result["tokens"].unbind(), data["tokens"].unbind(), strict=True):
        assert torch.equal(actual, expected)
    assert result.batch_size == data.batch_size
    return manager._num_threads


def test_merge_keeps_non_ray_driver_uninitialized(external_controller):
    manager = KVStorageManager(external_controller, {"client_name": "ExecutorTestMemoryClient"})
    try:
        assert not ray.is_initialized()
        threads = merge(manager)
        assert not ray.is_initialized()
        assert threads == min(max(2, os.cpu_count() or 2), LIMIT_THREADS_PER_MANAGER_IN_DRIVER)
    finally:
        manager.close()


def test_initialized_driver_and_actor_keep_resource_sizing():
    assert not ray.is_initialized()
    ray.init(address="local", num_cpus=4, num_gpus=0, object_store_memory=256 * 1024 * 1024)
    controller = TransferQueueController.remote()
    actor = None
    try:
        info = ray.get(controller.get_zmq_server_info.remote())
        manager = KVStorageManager(info, {"client_name": "RayStorageClient"})
        try:
            assert merge(manager) == min(max(2, os.cpu_count() or 2), LIMIT_THREADS_PER_MANAGER_IN_DRIVER)
        finally:
            manager.close()

        @ray.remote(num_cpus=2)
        class MergeActor:
            def check(self, controller_info):
                from transfer_queue.storage.managers.base import KVStorageManager

                manager = KVStorageManager(controller_info, {"client_name": "RayStorageClient"})
                try:
                    return merge(manager), ray.get_runtime_context().get_assigned_resources()["CPU"]
                finally:
                    manager.close()

        actor = MergeActor.options(
            runtime_env={
                "env_vars": {"PYTHONPATH": os.path.dirname(__file__) + os.pathsep + os.environ.get("PYTHONPATH", "")}
            }
        ).remote()
        threads, assigned = ray.get(actor.check.remote(info))
        assert assigned == 2
        assert threads == min(max(2, int(assigned)), LIMIT_THREADS_PER_MANAGER_IN_RAY_ACTOR)
        assert ray.is_initialized()
    finally:
        if actor is not None:
            ray.kill(actor)
        ray.kill(controller)
        ray.shutdown()
