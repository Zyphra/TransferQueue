# Copyright 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2025 The TransferQueue Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Real manager/client methods; scripted shard awaits, no sockets or services.

Constructors that start networking are bypassed. This is a local ownership
regression, not remote quiescence, physical cleanup, or restore qualification.
"""

import asyncio

import pytest
import torch
from tensordict import TensorDict

from transfer_queue.client import AsyncTransferQueueClient
from transfer_queue.metadata import BatchMeta
from transfer_queue.storage.managers.simple_storage_manager import AsyncSimpleStorageManager
from transfer_queue.storage.simple_storage import StorageUnitData


class ControlledShards(AsyncSimpleStorageManager):
    def __init__(self):
        # Use real routing, tensor slicing, schema extraction and storage data;
        # replace only the transport awaits with explicit asyncio events.
        self.storage_manager_id = "settlement-control"
        self.storage_unit_infos = {"first": None, "second": None}
        self.data = {key: StorageUnitData(None) for key in self.storage_unit_infos}
        self.started = {key: asyncio.Event() for key in self.storage_unit_infos}
        self.release = {key: asyncio.Event() for key in self.storage_unit_infos}
        self.ended = {key: asyncio.Event() for key in self.storage_unit_infos}
        self.errors = {}
        self.shard_cancellations = []
        self.notifications = []

    async def _operation(self, unit, apply):
        self.started[unit].set()
        try:
            await self.release[unit].wait()
            if unit in self.errors:
                raise self.errors[unit]
            apply()
        except asyncio.CancelledError as error:
            self.shard_cancellations.append((unit, error))
            raise
        finally:
            self.ended[unit].set()

    async def _put_to_single_storage_unit(self, indexes, fields, target_storage_unit, data_parser=None):
        assert data_parser is None
        await self._operation(
            target_storage_unit,
            lambda: self.data[target_storage_unit].put_data(fields, indexes),
        )

    async def _clear_single_storage_unit(self, indexes, target_storage_unit):
        await self._operation(
            target_storage_unit,
            lambda: self.data[target_storage_unit].clear(indexes),
        )

    async def notify_data_update(self, partition, indexes, schema):
        self.notifications.append((partition, indexes, schema))


class ControlledClient(AsyncTransferQueueClient):
    def __init__(self, manager):
        self.client_id = "settlement-control-client"
        self._controller = object()
        self.storage_manager = manager
        self.marked = []
        self.released = []

    async def _mark_clearing_in_controller(self, metadata, socket=None):
        self.marked.append(metadata)

    async def _clear_meta_in_controller(self, metadata, socket=None):
        self.released.append(metadata)


def metadata():
    return BatchMeta(global_indexes=[0, 1], partition_ids=["train", "train"])


def tensors():
    return TensorDict({"rm_scores": torch.tensor([[1.0], [2.0]])}, batch_size=[2])


async def started(manager):
    await asyncio.gather(*(event.wait() for event in manager.started.values()))


async def turn():
    # Run all currently-ready callbacks without a wall-clock timing assumption.
    for _ in range(6):
        await asyncio.sleep(0)


def assert_no_owned_tasks():
    assert asyncio.all_tasks() == {asyncio.current_task()}


def operation(manager, kind):
    if kind == "put":
        return manager.put_data(tensors(), metadata())
    return manager.clear_data(metadata())


def test_success_preserves_real_payload_and_notifies_once():
    async def run():
        manager = ControlledShards()
        task = asyncio.create_task(operation(manager, "put"))
        await started(manager)
        manager.release["first"].set()
        await manager.ended["first"].wait()
        await turn()
        assert not task.done()
        assert not manager.notifications
        manager.release["second"].set()
        await task
        for index, unit in enumerate(("first", "second")):
            value = manager.data[unit].get_data(["rm_scores"], [index])["rm_scores"][0]
            assert torch.equal(value, torch.tensor([float(index + 1)]))
        assert len(manager.notifications) == 1
        assert_no_owned_tasks()

    asyncio.run(run())


@pytest.mark.parametrize("kind", ["put", "clear"])
def test_failure_waits_for_sibling_and_preserves_original_exception(kind):
    async def run():
        manager = ControlledShards()
        error = RuntimeError("first shard failed")
        manager.errors["first"] = error
        task = asyncio.create_task(operation(manager, kind))
        await started(manager)
        manager.release["first"].set()
        await manager.ended["first"].wait()
        await turn()
        assert not task.done()
        manager.release["second"].set()
        with pytest.raises(RuntimeError) as caught:
            await task
        assert caught.value is error
        assert manager.ended["second"].is_set()
        assert not manager.notifications
        assert_no_owned_tasks()

    asyncio.run(run())


@pytest.mark.parametrize("kind", ["put", "clear"])
def test_first_failure_precedes_routing_order_and_repeated_cancellation(kind):
    async def run():
        manager = ControlledShards()
        first = ValueError("second route failed first")
        manager.errors = {"first": RuntimeError("later first route failure"), "second": first}
        task = asyncio.create_task(operation(manager, kind))
        await started(manager)
        manager.release["second"].set()
        await manager.ended["second"].wait()
        await turn()
        task.cancel("later cancellation one")
        await turn()
        task.cancel("later cancellation two")
        await turn()
        assert not task.done()
        manager.release["first"].set()
        with pytest.raises(ValueError) as caught:
            await task
        assert caught.value is first
        assert not manager.shard_cancellations
        assert_no_owned_tasks()

    asyncio.run(run())


@pytest.mark.parametrize("kind", ["put", "clear"])
def test_repeated_caller_cancellation_drains_shards_and_keeps_first_cause(kind):
    async def run():
        manager = ControlledShards()
        manager.errors["second"] = RuntimeError("secondary shard failure")
        task = asyncio.create_task(operation(manager, kind))
        await started(manager)
        task.cancel("original caller cancellation")
        await turn()
        task.cancel("repeated caller cancellation")
        await turn()
        assert not task.done()
        assert not manager.shard_cancellations
        for event in manager.release.values():
            event.set()
        with pytest.raises(asyncio.CancelledError) as caught:
            await task
        assert caught.value.args == ("original caller cancellation",)
        assert all(event.is_set() for event in manager.ended.values())
        assert not manager.shard_cancellations
        assert not manager.notifications
        assert_no_owned_tasks()

    asyncio.run(run())


@pytest.mark.parametrize("kind", ["put", "clear"])
def test_cancelled_shard_is_failure_and_sibling_settles(kind):
    async def run():
        manager = ControlledShards()
        error = asyncio.CancelledError("shard cancellation")
        manager.errors["first"] = error
        task = asyncio.create_task(operation(manager, kind))
        await started(manager)
        manager.release["first"].set()
        await manager.ended["first"].wait()
        await turn()
        assert not task.done()
        manager.release["second"].set()
        with pytest.raises(asyncio.CancelledError) as caught:
            await task
        assert caught.value is error
        assert manager.ended["second"].is_set()
        assert not manager.notifications
        assert_no_owned_tasks()

    asyncio.run(run())


def test_failed_clear_retains_client_metadata_then_successful_retry_releases():
    async def run():
        manager = ControlledShards()
        for index, unit in enumerate(manager.storage_unit_infos):
            manager.data[unit].put_data({"rm_scores": [torch.tensor([float(index)])]}, [index])
        client = ControlledClient(manager)
        meta = metadata()
        error = RuntimeError("storage clear failed")
        manager.errors["first"] = error
        task = asyncio.create_task(client.async_clear_samples(meta))
        await started(manager)
        manager.release["first"].set()
        await manager.ended["first"].wait()
        await turn()
        assert not task.done()
        assert not client.released
        manager.release["second"].set()
        with pytest.raises(RuntimeError) as caught:
            await task
        assert caught.value.__cause__ is error
        assert client.marked == [meta]
        assert not client.released
        assert manager.data["first"].active_key_count == 1
        assert manager.data["second"].active_key_count == 0
        manager.errors.clear()
        await client.async_clear_samples(meta)
        assert client.released == [meta]
        assert all(unit.active_key_count == 0 for unit in manager.data.values())
        assert_no_owned_tasks()

    asyncio.run(run())


def test_cancelled_clear_retains_client_metadata_through_settlement():
    async def run():
        manager = ControlledShards()
        client = ControlledClient(manager)
        task = asyncio.create_task(client.async_clear_samples(metadata()))
        await started(manager)
        task.cancel("clear interrupted")
        await turn()
        assert not task.done()
        assert not client.released
        for event in manager.release.values():
            event.set()
        with pytest.raises(asyncio.CancelledError) as caught:
            await task
        assert caught.value.args == ("clear interrupted",)
        assert not client.released
        assert all(event.is_set() for event in manager.ended.values())
        assert_no_owned_tasks()

    asyncio.run(run())
