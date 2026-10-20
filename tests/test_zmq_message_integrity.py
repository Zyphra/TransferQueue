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

"""Frame checksums: detection on decode and bounded re-reads through a real storage unit."""

import threading
import time

import numpy as np
import pytest
import ray
import torch
import zmq
import zmq.asyncio
from tensordict import TensorDict

from transfer_queue.metadata import BatchMeta
from transfer_queue.storage import AsyncSimpleStorageManager
from transfer_queue.storage.simple_storage import SimpleStorageUnit
from transfer_queue.utils.enum_utils import Role
from transfer_queue.utils.zmq_utils import (
    ZMQMessage,
    ZMQMessageIntegrityError,
    ZMQRequestType,
    ZMQServerInfo,
    integrity_counts,
)


def _flip_byte(frame) -> bytes:
    """A copy of ``frame`` with one byte in the middle inverted."""
    data = bytearray(memoryview(frame).tobytes())
    data[len(data) // 2] ^= 0xFF
    return bytes(data)


def _get_response(rows: int = 4) -> ZMQMessage:
    data = {
        "f": [torch.arange(256, dtype=torch.float32) + i for i in range(rows)],
        "ids": [torch.arange(64, dtype=torch.int64) * i for i in range(rows)],
    }
    return ZMQMessage.create(ZMQRequestType.GET_DATA_RESPONSE, sender_id="unit", body={"data": data})


def test_clean_message_round_trips_without_counting():
    before = integrity_counts()
    frames = _get_response().serialize()
    decoded = ZMQMessage.deserialize([memoryview(f).tobytes() for f in frames])
    assert torch.equal(decoded.body["data"]["f"][2], torch.arange(256, dtype=torch.float32) + 2)
    assert integrity_counts() == before


def test_corrupt_tensor_frame_names_field_and_row():
    before = integrity_counts()
    frames = [memoryview(f).tobytes() for f in _get_response().serialize()]
    # Frame order follows encoding: header, then rows of "f", then rows of "ids".
    frames[1 + 4 + 3] = _flip_byte(frames[1 + 4 + 3])
    with pytest.raises(ZMQMessageIntegrityError) as raised:
        ZMQMessage.deserialize(frames)
    assert raised.value.locations == [("body", "data", "ids", 3)]
    assert integrity_counts()["checksum_mismatches"] == before["checksum_mismatches"] + 1


def test_corrupt_header_or_missing_checksum_is_rejected():
    frames = [memoryview(f).tobytes() for f in _get_response().serialize()]
    with pytest.raises(ZMQMessageIntegrityError) as raised:
        ZMQMessage.deserialize([_flip_byte(frames[0]), *frames[1:]])
    assert raised.value.locations == [("frame", 0)]
    with pytest.raises(ZMQMessageIntegrityError, match="checksum frame missing"):
        ZMQMessage.deserialize(frames[:-1])


class _CorruptingRelay:
    """A real ZMQ hop between one client and a storage unit that can damage chosen messages.

    It flips one byte in the largest buffer frame (never the header or checksum) of the next
    ``corrupt_requests`` requests and ``corrupt_responses`` responses, like a transport fault.
    """

    def __init__(self, unit_address: str):
        self.context = zmq.Context()
        self.front = self.context.socket(zmq.ROUTER)
        self.port = self.front.bind_to_random_port("tcp://127.0.0.1")
        self.back = self.context.socket(zmq.DEALER)
        self.back.connect(unit_address)
        self.corrupt_requests = 0
        self.corrupt_responses = 0
        self.requests_seen = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    @staticmethod
    def _damage(frames: list) -> list:
        frames = [memoryview(f).tobytes() for f in frames]
        target = max(range(1, len(frames) - 1), key=lambda i: len(frames[i]))
        frames[target] = _flip_byte(frames[target])
        return frames

    def _run(self):
        poller = zmq.Poller()
        poller.register(self.front, zmq.POLLIN)
        while not self._stop.is_set():
            if not poller.poll(100):
                continue
            client, *request = self.front.recv_multipart()
            self.requests_seen += 1
            if self.corrupt_requests:
                self.corrupt_requests -= 1
                request = self._damage(request)
            self.back.send_multipart(request)
            response = self.back.recv_multipart()
            if self.corrupt_responses:
                self.corrupt_responses -= 1
                response = self._damage(response)
            self.front.send_multipart([client, *response])

    def close(self):
        self._stop.set()
        self._thread.join()
        self.context.destroy(linger=0)


class _ManagerWithoutController(AsyncSimpleStorageManager):
    """The real manager data path; the controller notification is outside this test."""

    async def notify_data_update(self, *args, **kwargs):
        return None


@pytest.fixture(scope="module")
def relay_and_manager():
    ray.init(address="local", num_cpus=2, num_gpus=0, object_store_memory=256 * 1024 * 1024, ignore_reinit_error=True)
    unit = SimpleStorageUnit.options(num_cpus=1).remote(storage_unit_size=1000)
    info = ray.get(unit.get_zmq_server_info.remote())
    time.sleep(1)  # the unit's proxy thread starts after its handle is returned
    relay = _CorruptingRelay(info.to_addr("put_get_socket"))
    manager = _ManagerWithoutController.__new__(_ManagerWithoutController)
    manager.storage_manager_id = "integrity_test_manager"
    manager.zmq_context = zmq.asyncio.Context()
    manager.storage_unit_infos = {
        info.id: ZMQServerInfo(role=Role.STORAGE, id=info.id, ip="127.0.0.1", ports={"put_get_socket": relay.port})
    }
    yield relay, manager, info.id
    relay.close()
    ray.kill(unit)
    ray.shutdown()


def _same(result, data) -> bool:
    """Gets return per-row (nested) tensors; compare them row by row with what was put."""
    return all(torch.equal(torch.stack(result[name].unbind()), data[name]) for name in ("f", "ids"))


def _batch(rows: int = 4):
    data = {
        "f": torch.randn(rows, 4096),
        "ids": torch.randint(0, 50_000, (rows, 1024), dtype=torch.int64),
    }
    schema = {
        name: {"dtype": value.dtype, "shape": tuple(value.shape[1:]), "is_nested": False, "is_non_tensor": False}
        for name, value in data.items()
    }
    meta = BatchMeta(
        global_indexes=list(range(rows)),
        partition_ids=["p0"] * rows,
        field_schema=schema,
        production_status=np.ones(rows, dtype=np.int8),
    )
    return TensorDict(data, batch_size=rows), meta


@pytest.mark.asyncio
async def test_clean_put_get_through_storage_unit(relay_and_manager):
    relay, manager, _ = relay_and_manager
    data, meta = _batch()
    before = integrity_counts()
    await manager.put_data(data, meta)
    result = await manager.get_data(meta)
    assert _same(result, data)
    assert integrity_counts() == before


@pytest.mark.asyncio
async def test_corrupt_get_response_is_reread(relay_and_manager):
    relay, manager, _ = relay_and_manager
    data, meta = _batch()
    await manager.put_data(data, meta)
    before, seen = integrity_counts(), relay.requests_seen
    relay.corrupt_responses = 2
    result = await manager.get_data(meta)
    assert _same(result, data)
    assert relay.requests_seen - seen == 3
    after = integrity_counts()
    assert after["checksum_mismatches"] - before["checksum_mismatches"] == 2
    assert after["retry_successes"] - before["retry_successes"] == 1


@pytest.mark.asyncio
async def test_corrupt_put_request_is_resent(relay_and_manager):
    relay, manager, _ = relay_and_manager
    data, meta = _batch()
    before = integrity_counts()
    relay.corrupt_requests = 1
    await manager.put_data(data, meta)
    result = await manager.get_data(meta)
    assert _same(result, data)
    after = integrity_counts()
    assert after["checksum_mismatches"] - before["checksum_mismatches"] == 1
    assert after["retry_successes"] - before["retry_successes"] == 1


@pytest.mark.asyncio
async def test_persistent_corruption_fails_naming_unit_field_and_index(relay_and_manager):
    relay, manager, unit_id = relay_and_manager
    data, meta = _batch()
    await manager.put_data(data, meta)
    seen = relay.requests_seen
    relay.corrupt_responses = 100
    try:
        with pytest.raises(ZMQMessageIntegrityError) as raised:
            await manager.get_data(meta)
    finally:
        relay.corrupt_responses = 0
    message = str(raised.value)
    assert unit_id in message and "all 4 attempts" in message
    assert "field='f' global_index=" in message
    assert relay.requests_seen - seen == 4
