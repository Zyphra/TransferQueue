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

import copy
import json
import math
import os
import shutil
import subprocess
import time
from concurrent.futures import Future
from importlib import resources
from pathlib import Path
from threading import Lock, Thread
from typing import Any, Callable, cast

import ray
import torch
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from tensordict.tensorclass import NonTensorStack

from transfer_queue.client import TransferQueueClient
from transfer_queue.controller import TransferQueueController
from transfer_queue.metadata import KVBatchMeta
from transfer_queue.sampler import *  # noqa: F401
from transfer_queue.sampler import BaseSampler
from transfer_queue.storage.bootstrap import StorageBootstrapProvider
from transfer_queue.storage.managers.simple_storage_manager import AsyncSimpleStorageManager
from transfer_queue.utils.logging_utils import get_logger
from transfer_queue.utils.yuanrong_utils import cleanup_yuanrong_resources
from transfer_queue.utils.zmq_utils import process_zmq_server_info

logger = get_logger(__name__)

_TQ_CLIENT: Any = None
_TQ_STORAGE: Any = None
_TQ_CONTROLLER: Any = None


def _maybe_create_tq_client(conf: DictConfig | None = None) -> TransferQueueClient:
    global _TQ_CLIENT
    if _TQ_CLIENT is None:
        if conf is None:
            _init_from_existing()
            assert _TQ_CLIENT is not None, (
                "TransferQueueController has not been initialized yet. Please call init() first."
            )
            return _TQ_CLIENT

        pid = os.getpid()
        _TQ_CLIENT = TransferQueueClient(
            client_id=f"TransferQueueClient_{pid}", controller_info=conf.controller.zmq_info
        )

        backend_name = conf.backend.storage_backend

        _TQ_CLIENT.initialize_storage_manager(manager_type=backend_name, config=conf.backend[backend_name])

    return _TQ_CLIENT


def _maybe_create_tq_storage(conf: DictConfig) -> DictConfig:
    global _TQ_STORAGE

    if _TQ_STORAGE is None:
        _TQ_STORAGE = {}
        backend_name = conf.backend.storage_backend
        provider_fn = StorageBootstrapProvider.get_provider(backend_name)
        if provider_fn is not None:
            backend_resources = provider_fn(conf)
            if backend_resources is not None:
                _TQ_STORAGE[backend_name] = backend_resources
            else:
                logger.error(f"Not found available {backend_name} storage resources, please check the config.")
        else:
            logger.error(
                f"Storage backend {backend_name} not registered. Please add it to the StorageBootstrapProvider."
            )
    return conf


def _init_from_existing() -> bool:
    """Initialize the TransferQueueClient from existing controller.

    Returns:
        True if successfully initialized from existing controller, False otherwise.
    """
    global _TQ_CONTROLLER
    try:
        if _TQ_CONTROLLER is None:
            _TQ_CONTROLLER = ray.get_actor("TransferQueueController", namespace="transfer_queue")

    except ValueError:
        logger.info("Called _init_from_existing() but TransferQueueController has not been initialized yet.")
        return False

    logger.info("Found existing TransferQueueController instance. Connecting...")

    conf = None
    while conf is None:
        conf = ray.get(_TQ_CONTROLLER.get_config.remote())
        if conf is not None:
            _maybe_create_tq_client(conf)

            logger.info("TransferQueueClient initialized.")
            return True

        logger.debug("Waiting for controller to initialize... Retrying in 1s")
        time.sleep(1)

    return False


# ==================== Initialization API ====================
def init(conf: DictConfig | None = None) -> DictConfig | None:
    """Initialize the TransferQueue system.

    This function sets up the TransferQueue controller, distributed storage, and client.
    It should be called once at the beginning of the program before any data operations.

    If a controller already exists, reuse it and only initialize the client;
    the provided `conf` will be ignored in this case.

    Args:
        conf: Optional custom config merged with default `config.yaml`.
              Only takes effect on first-time initialization, ignored when attaching
              to an existing controller.
    Returns:
        The merged configuration dictionary.

    Example:
        >>> # In process 0, node A
        >>> import transfer_queue as tq
        >>> tq.init()   # Initialize the TransferQueue
        >>> tq.put(...) # then you can use tq for data operations
        >>>
        >>> # In process 1, node B (with Ray connected to node A)
        >>> import transfer_queue as tq
        >>> tq.init()   # This will only initialize a TransferQueueClient and link with existing TQ
        >>> metadata = tq.get_meta(...)
        >>> data = tq.get_data(metadata)
    """
    if _init_from_existing():
        return conf

    logger.info("No TransferQueueController found. Starting first-time initialization...")

    final_conf = OmegaConf.create({}, flags={"allow_objects": True})
    default_conf = OmegaConf.load(resources.files("transfer_queue") / "config.yaml")
    final_conf = OmegaConf.merge(final_conf, default_conf)
    if conf:
        final_conf = OmegaConf.merge(final_conf, conf)

    # TODO(hz): support load custom sampler class from external module.
    try:
        sampler = final_conf.controller.sampler
        if isinstance(sampler, BaseSampler):
            # user pass a pre-initialized sampler instance
            sampler = sampler
        elif isinstance(sampler, type) and issubclass(sampler, BaseSampler):
            # user pass a sampler class
            sampler = sampler()
        elif isinstance(sampler, str):
            # user pass a sampler name str
            # try to convert as sampler class
            sampler = globals()[final_conf.controller.sampler]
    except KeyError:
        raise ValueError(f"Could not find sampler {final_conf.controller.sampler}") from None

    try:
        global _TQ_CONTROLLER
        _TQ_CONTROLLER = TransferQueueController.options(  # type: ignore[attr-defined]
            name="TransferQueueController", namespace="transfer_queue"
        ).remote(sampler=sampler, polling_mode=final_conf.controller.polling_mode)
        logger.info("TransferQueueController has been created.")
    except ValueError:
        logger.info("Some other rank has initialized TransferQueueController. Try to connect to existing controller.")
        _init_from_existing()
        return final_conf

    controller_zmq_info = process_zmq_server_info(_TQ_CONTROLLER)
    final_conf.controller.zmq_info = controller_zmq_info

    final_conf = _maybe_create_tq_storage(final_conf)

    ray.get(_TQ_CONTROLLER.store_config.remote(final_conf))
    logger.info(f"TransferQueue config: {final_conf}")

    # start Prometheus metrics exporter if enabled
    metrics_conf = final_conf.get("metrics", {})
    if metrics_conf.get("enabled", False):
        metrics_port = metrics_conf.get("port", 0)
        metrics_endpoint = ray.get(_TQ_CONTROLLER.start_metrics.remote(port=metrics_port))
        final_conf.metrics.enabled = True
        final_conf.metrics.endpoint = metrics_endpoint
        # Update stored config so other processes can discover the endpoint
        ray.get(_TQ_CONTROLLER.store_config.remote(final_conf))
        # Register storage units for metrics collection (SimpleStorage only)
        if final_conf.backend.storage_backend == "SimpleStorage":
            storage_zmq_info = final_conf.backend.SimpleStorage.get("zmq_info")
            if storage_zmq_info:
                ray.get(_TQ_CONTROLLER.register_storage_units_for_metrics.remote(storage_zmq_info))
            # Start metrics exporter on each storage unit
            if _TQ_STORAGE and "SimpleStorage" in _TQ_STORAGE:
                futures = [handle.start_metrics.remote(port=0) for handle in _TQ_STORAGE["SimpleStorage"].values()]
                ray.get(futures)
        logger.info(f"Prometheus metrics endpoint: http://{metrics_endpoint}/metrics")

    _maybe_create_tq_client(final_conf)
    return final_conf


def close():
    """Close the TransferQueue system.

    This function cleans up the TransferQueue system, including:
    - Closing the client and its associated resources
    - Cleaning up distributed storage (only for the process that initialized it)
    - Killing the controller actor

    Note:
        This function should be called when the TransferQueue system is no longer needed.
    """
    global _TQ_CLIENT
    global _TQ_STORAGE
    global _TQ_CONTROLLER
    global _PENDING_CHECKPOINT
    if _PENDING_CHECKPOINT is not None and not _PENDING_CHECKPOINT.done():
        raise RuntimeError("join the pending checkpoint Future before closing TransferQueue")
    _PENDING_CHECKPOINT = None

    try:
        if _TQ_STORAGE:
            for key, value in _TQ_STORAGE.items():
                if key == "SimpleStorage":
                    # only the process that do first-time init can clean the distributed storage
                    for storage in value.values():
                        ray.kill(storage)
                elif key == "MooncakeStore":
                    check = subprocess.run(["pgrep", "-f", "mooncake_master"], stdout=subprocess.PIPE, text=True)
                    if check.returncode == 0:
                        pids = check.stdout.strip().replace("\n", ", ")
                        logger.warning(
                            f"TransferQueue will not stop mooncake_master process with PID: {pids}. "
                            f"Consider manually killing the mooncake_master."
                        )

                    # Terminate offload client process if it was started
                    if isinstance(value, dict):
                        offload_proc = value.get("offload_client_process")
                        if offload_proc is not None and offload_proc.poll() is None:
                            offload_proc.terminate()
                            try:
                                offload_proc.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                offload_proc.kill()
                            logger.info(f"Terminated mooncake_client offload process (PID: {offload_proc.pid}).")

                    if _TQ_CLIENT:
                        try:
                            ret = _TQ_CLIENT.storage_manager.storage_client._store.remove_all()
                            if ret < 0:
                                logger.error("Failed to remove existing keys in mooncake_master.")
                            else:
                                logger.info("Successfully removed all existing keys in mooncake_master.")
                        except Exception:
                            pass
                elif key == "Yuanrong":
                    cleanup_yuanrong_resources(value)
                else:
                    logger.warning(f"close for _TQ_STORAGE with key {key} is not supported for now.")

        _TQ_STORAGE = None
    except Exception:
        pass

    if _TQ_CLIENT:
        _TQ_CLIENT.close()
        _TQ_CLIENT = None

    if _TQ_CONTROLLER:
        try:
            ray.kill(_TQ_CONTROLLER)
        except Exception:
            pass
        _TQ_CONTROLLER = None


# ==================== Metrics API ====================
def get_metrics_endpoint() -> str | None:
    """Return the Prometheus metrics endpoint address (``host:port``), or *None* if metrics are disabled.

    Works from any process — the endpoint is stored in the Controller's config
    so that processes joining via ``_init_from_existing()`` can discover it too.

    Example:
        >>> import transfer_queue as tq
        >>> tq.init({"metrics": {"enabled": True}})
        >>> endpoint = tq.get_metrics_endpoint()
        >>> print(endpoint)   # e.g. "10.0.1.42:38271"
        >>> # Use endpoint to register Prometheus scrape target
    """
    if _TQ_CONTROLLER is None:
        _init_from_existing()
    if _TQ_CONTROLLER is None:
        return None
    conf = ray.get(_TQ_CONTROLLER.get_config.remote())
    if conf is None:
        return None
    return conf.get("metrics", {}).get("endpoint", None)


# ==================== High-Level KV Interface API ====================
def kv_put(
    key: str,
    partition_id: str,
    fields: TensorDict | dict[str, Any] | None = None,
    tag: dict[str, Any] | None = None,
    data_parser: Callable[[Any], Any] | None = None,
) -> KVBatchMeta:
    """Put a single key-value pair to TransferQueue.

    This is a convenience method for putting data using a user-specified key
    instead of BatchMeta. Internally, the key is translated to a BatchMeta
    and the data is stored using the regular put mechanism.

    Args:
        key: User-specified key for the data sample (in row)
        partition_id: Logical partition to store the data in
        fields: Data fields to store. Can be a TensorDict or a dict of tensors.
                Each key in `fields` will be treated as a column for the data sample.
                If dict is provided, tensors will be unsqueezed to add batch dimension.
                If not provided, will only update the newly given tag to the key.
        tag: Optional metadata tag to associate with the key
        data_parser: Optional callable to parse reference data (e.g., URLs) into real
                     content. The input is a slice of the `fields` parameter passed to
                     kv_put / kv_batch_put, in plain dict format (not TensorDict),
                     mapping field_name -> batched values. For a regular tensor column
                     the value is a batched tensor; for nested tensors (jagged or
                     strided) and NonTensorStack columns the values are extracted into
                     a list. It must modify values in-place based on the original keys;
                     do not add or remove keys. The number of elements per column must
                     also remain unchanged. Do not change the inner order of values
                     within each column. Only supported by SimpleStorage.

    Returns:
        KVBatchMeta: Metadata containing the key, tags, partition_id, and fields.
                     The `fields` attribute includes all fields stored for this sample,
                     including any new fields written by this put operation.

    Raises:
        ValueError: If neither fields nor tag is provided
        ValueError: If nested tensors are provided (use kv_batch_put instead)
        RuntimeError: If retrieved BatchMeta size doesn't match length of `keys`

    Example:
        >>> import transfer_queue as tq
        >>> import torch
        >>> tq.init()
        >>> # Put with both fields and tag
        >>> meta = tq.kv_put(
        ...     key="sample_1",
        ...     partition_id="train",
        ...     fields={"input_ids": torch.tensor([1, 2, 3])},
        ...     tag={"score": 0.95}
        ... )
        >>> print(meta.fields)  # ['input_ids']
    """
    tq_client = _maybe_create_tq_client()
    return tq_client._run_coroutine(
        async_kv_put(
            key=key,
            partition_id=partition_id,
            fields=fields,
            tag=tag,
            data_parser=data_parser,
        )
    )


def kv_batch_put(
    keys: list[str],
    partition_id: str,
    fields: TensorDict | None = None,
    tags: list[dict[str, Any]] | None = None,
    data_parser: Callable[[Any], Any] | None = None,
) -> KVBatchMeta:
    """Batch put multiple key-value pairs into the TransferQueue.

    This method stores multiple key-value entries in a single operation,
    which is significantly more efficient than repeated calls to ``kv_put``.

    Args:
        keys: List of user-defined unique keys for the data entries.
        partition_id: Logical partition where the data will be stored.
        fields: TensorDict containing batched data for all keys. Must have ``batch_size == len(keys)``.
            If not provided, only the associated tags will be updated.
        tags: List of metadata dictionaries, one per key.  Length must match the number of keys.
        data_parser: Optional callable to parse raw reference data (e.g., URLs) into real content
            before storage. The input is a plain dict (not TensorDict) mapping field names to
            batched values. The parser  **must modify data in-place** without adding/removing
            keys or changing element counts/order. Only supported by ``SimpleStorage`` backend.

    Returns:
        KVBatchMeta: Metadata object containing stored keys, tags, partition ID,
            and field information. The ``fields`` attribute includes all
            persisted fields for the written samples.

    Raises:
        ValueError: When both ``fields`` and ``tags`` are empty.
        ValueError: When ``fields`` batch size mismatches key count.
        ValueError: When ``tags`` length mismatches key count.
        RuntimeError: When retrieved metadata size mismatches input key count.

    Example:
        >>> import transfer_queue as tq
        >>> from tensordict import TensorDict
        >>> tq.init()
        >>> keys = ["sample_1", "sample_2", "sample_3"]
        >>> fields = TensorDict({
        ...     "input_ids": torch.randn(3, 10),
        ...     "attention_mask": torch.ones(3, 10),
        ... }, batch_size=3)
        >>> tags = [{"score": 0.9}, {"score": 0.85}, {"score": 0.95}]
        >>> meta = tq.kv_batch_put(keys=keys, partition_id="train", fields=fields, tags=tags)
        >>> print(meta.fields)
    """
    tq_client = _maybe_create_tq_client()
    return tq_client._run_coroutine(
        async_kv_batch_put(
            keys=keys,
            partition_id=partition_id,
            fields=fields,
            tags=tags,
            data_parser=data_parser,
        )
    )


def kv_batch_get_by_meta(meta: KVBatchMeta, select_fields: list[str] | str | None = None) -> TensorDict:
    """Get data from TransferQueue using KVBatchMeta.

    This is a convenience method for retrieving data using KVBatchMeta returned
    from a previous put operation. It extracts the keys and partition_id from
    the metadata to fetch the corresponding data.

    Args:
        meta: KVBatchMeta object returned from a previous put operation (e.g., kv_put,
              kv_batch_put). It contains keys, partition_id, and fields information.
        select_fields: Optional field(s) to retrieve, which overrides the fields
                       recorded in the given KVBatchMeta. If None, uses all fields
                       from meta.fields. Can be a single field name (str) or a list
                       of field names.

    Returns:
        TensorDict with the requested data

    Raises:
        ValueError: If keys or partition are not found
        ValueError: If empty fields exist in any key (sample)
        ValueError: If any field in select_fields doesn't exist in KVBatchMeta.fields

    Example:
        >>> import transfer_queue as tq
        >>> tq.init()
        >>> # First put some data
        >>> keys = ["sample_1", "sample_2", "sample_3"]
        >>> fields = TensorDict({
        ...     "input_ids": torch.randn(3, 10),
        ...     "attention_mask": torch.ones(3, 10),
        ... }, batch_size=3)
        >>> meta = tq.kv_batch_put(keys=keys, partition_id="train", fields=fields)
        >>> # Then retrieve it using the returned metadata
        >>> data = tq.kv_batch_get_by_meta(meta)
    """
    tq_client = _maybe_create_tq_client()
    return tq_client._run_coroutine(async_kv_batch_get_by_meta(meta=meta, select_fields=select_fields))


def kv_batch_get(keys: list[str] | str, partition_id: str, select_fields: list[str] | str | None = None) -> TensorDict:
    """Get data from TransferQueue using user-specified keys.

    This is a convenience method for retrieving data using keys instead of indexes.

    Args:
        keys: Single key or list of keys to retrieve
        partition_id: Partition containing the keys
        select_fields: Optional field(s) to retrieve. If None, retrieves all fields

    Returns:
        TensorDict with the requested data

    Raises:
        ValueError: If keys or partition are not found
        ValueError: If empty fields exist in any key (sample)

    Example:
        >>> import transfer_queue as tq
        >>> tq.init()
        >>> # Get single key with all fields
        >>> data = tq.kv_batch_get(keys="sample_1", partition_id="train")
        >>> # Get multiple keys with specific fields
        >>> data = tq.kv_batch_get(
        ...     keys=["sample_1", "sample_2"],
        ...     partition_id="train",
        ...     select_fields="input_ids"
        ... )
    """
    tq_client = _maybe_create_tq_client()
    return tq_client._run_coroutine(
        async_kv_batch_get(keys=keys, partition_id=partition_id, select_fields=select_fields)
    )


def kv_list(partition_id: str | None = None) -> dict[str, dict[str, Any]]:
    """List all keys and their metadata in one or all partitions.

    Args:
        partition_id: The specific partition_id to query.
            If None (default), returns keys from all partitions.

    Returns:
        A nested dictionary mapping partition IDs to their keys and metadata.

        Structure:
        {
            "partition_id": {
                "key_name": {
                    "tag1": <value>,
                    ... (other metadata)
                },
                ...,
            },
            ...
        }

    Example:
        >>> import transfer_queue as tq
        >>> tq.init()
        >>> # Case 1: Retrieve a specific partition
        >>> partitions = tq.kv_list(partition_id="train")
        >>> print(f"Keys: {list(partitions['train'].keys())}")
        >>> print(f"Tags: {list(partitions['train'].values())}")
        >>> # Case 2: Retrieve all partitions
        >>> all_partitions = tq.kv_list()
        >>> for pid, keys in all_partitions.items():
        >>>     print(f"Partition: {pid}, Key count: {len(keys)}")
    """
    tq_client = _maybe_create_tq_client()
    return tq_client._run_coroutine(async_kv_list(partition_id=partition_id))


def kv_clear(keys: list[str] | str, partition_id: str) -> None:
    """Clear key-value pairs from TransferQueue.

    This removes the specified keys and their associated data from both
    the controller and storage units.

    Args:
        keys: Single key or list of keys to clear
        partition_id: Partition containing the keys

    Example:
        >>> import transfer_queue as tq
        >>> tq.init()
        >>> # Clear single key
        >>> tq.kv_clear(keys="sample_1", partition_id="train")
        >>> # Clear multiple keys
        >>> tq.kv_clear(keys=["sample_1", "sample_2"], partition_id="train")
    """

    tq_client = _maybe_create_tq_client()
    tq_client._run_coroutine(async_kv_clear(keys=keys, partition_id=partition_id))


# ==================== KV Interface API ====================
async def async_kv_put(
    key: str,
    partition_id: str,
    fields: TensorDict | dict[str, Any] | None = None,
    tag: dict[str, Any] | None = None,
    data_parser: Callable[[Any], Any] | None = None,
) -> KVBatchMeta:
    """Asynchronously put a single key-value pair to TransferQueue.

    This is a convenience method for putting data using a user-specified key
    instead of BatchMeta. Internally, the key is translated to a BatchMeta
    and the data is stored using the regular put mechanism.

    Args:
        key: User-specified key for the data sample (in row)
        partition_id: Logical partition to store the data in
        fields: Data fields to store. Can be a TensorDict or a dict of tensors.
                Each key in `fields` will be treated as a column for the data sample.
                If dict is provided, tensors will be unsqueezed to add batch dimension.
                If not provided, will only update the newly given tag to the key.
        tag: Optional metadata tag to associate with the key
        data_parser: Optional callable to parse reference data (e.g., URLs) into real
                     content. The input is a slice of the `fields` parameter passed to
                     kv_put / kv_batch_put, in plain dict format (not TensorDict),
                     mapping field_name -> batched values. For a regular tensor column
                     the value is a batched tensor; for nested tensors (jagged or
                     strided) and NonTensorStack columns the values are extracted into
                     a list. It must modify values in-place based on the original keys;
                     do not add or remove keys. The number of elements per column must
                     also remain unchanged. Do not change the inner order of values
                     within each column. Only supported by SimpleStorage.

    Returns:
        KVBatchMeta: Metadata containing the key, tags, partition_id, and fields.
                     The `fields` attribute includes all fields stored for this sample,
                     including any new fields written by this put operation.

    Raises:
        ValueError: If neither fields nor tag is provided
        ValueError: If nested tensors are provided (use kv_batch_put instead)
        RuntimeError: If retrieved BatchMeta size doesn't match length of `keys`

    Example:
        >>> import transfer_queue as tq
        >>> import torch
        >>> tq.init()
        >>> # Put with both fields and tag
        >>> meta = await tq.async_kv_put(
        ...     key="sample_1",
        ...     partition_id="train",
        ...     fields={"input_ids": torch.tensor([1, 2, 3])},
        ...     tag={"score": 0.95}
        ... )
        >>> print(meta.fields)  # ['input_ids']
    """

    if fields is None and tag is None:
        raise ValueError("Please provide at least one parameter of fields or tag.")

    tq_client = _maybe_create_tq_client()

    # 1. translate user-specified key to BatchMeta
    batch_meta = await tq_client.async_kv_retrieve_meta(keys=[key], partition_id=partition_id, create=True)

    if batch_meta.size != 1:
        raise RuntimeError(f"Retrieved BatchMeta size {batch_meta.size} does not match with input `key` size of 1!")

    # 2. register the user-specified tag to BatchMeta
    if tag is not None:
        batch_meta.update_custom_meta([tag])

    # 3. put data
    if fields is not None:
        if isinstance(fields, dict):
            # TODO: consider whether to support this...
            batch = {}
            for field_name, value in fields.items():
                if isinstance(value, torch.Tensor):
                    if value.is_nested:
                        raise ValueError("Please use (async)kv_batch_put for batch operation")
                    batch[field_name] = value.unsqueeze(0)
                else:
                    batch[field_name] = NonTensorStack(value)
            fields = TensorDict(batch, batch_size=[1])
        elif not isinstance(fields, TensorDict):
            raise ValueError("`fields` can only be dict or TensorDict")

        # After put, batch_meta.field_names will include the new fields written by user
        # A fields-only write must not republish an older snapshot of the tags.
        batch_meta = await tq_client._async_put(
            fields, batch_meta, data_parser=data_parser, publish_custom_meta=tag is not None
        )
    else:
        # Directly update custom_meta (tag) to controller
        await tq_client.async_set_custom_meta(batch_meta)

    fields_to_return = batch_meta.field_names

    return KVBatchMeta(
        keys=[key],
        tags=batch_meta.custom_meta,
        partition_id=partition_id,
        fields=fields_to_return,
        extra_info=batch_meta.extra_info,
    )


async def async_kv_batch_put(
    keys: list[str],
    partition_id: str,
    fields: TensorDict | None = None,
    tags: list[dict[str, Any]] | None = None,
    data_parser: Callable[[Any], Any] | None = None,
) -> KVBatchMeta:
    """Asynchronously put multiple key-value pairs to TransferQueue in batch.

    This method stores multiple key-value pairs in a single operation, which is more
    efficient than calling kv_put multiple times.

    Args:
        keys: List of user-specified keys for the data
        partition_id: Logical partition to store the data in
        fields: TensorDict containing data for all keys. Must have batch_size == len(keys).
                If not provided, will only update the newly given tags to the keys.
        tags: List of metadata tags, one for each key
        data_parser: Optional callable to parse reference data (e.g., URLs) into real
                     content. The input is a slice of the `fields` parameter passed to
                     kv_put / kv_batch_put, in plain dict format (not TensorDict),
                     mapping field_name -> batched values. For a regular tensor column
                     the value is a batched tensor; for nested tensors (jagged or
                     strided) and NonTensorStack columns the values are extracted into
                     a list. It must modify values in-place based on the original keys;
                     do not add or remove keys. The number of elements per column must
                     also remain unchanged. Do not change the inner order of values
                     within each column. Only supported by SimpleStorage.

    Returns:
        KVBatchMeta: Metadata containing the keys, tags, partition_id, and fields.
                     The `fields` attribute includes all fields stored for these samples,
                     including any new fields written by this put operation.

    Raises:
        ValueError: If neither `fields` nor `tags` is provided
        ValueError: If length of `keys` doesn't match length of `tags` or the batch_size of `fields` TensorDict
        RuntimeError: If retrieved BatchMeta size doesn't match length of `keys`

    Example:
        >>> import transfer_queue as tq
        >>> tq.init()
        >>> keys = ["sample_1", "sample_2", "sample_3"]
        >>> fields = TensorDict({
        ...     "input_ids": torch.randn(3, 10),
        ...     "attention_mask": torch.ones(3, 10),
        ... }, batch_size=3)
        >>> tags = [{"score": 0.9}, {"score": 0.85}, {"score": 0.95}]
        >>> meta = await tq.async_kv_batch_put(keys=keys, partition_id="train", fields=fields, tags=tags)
        >>> print(meta.fields)  # ['input_ids', 'attention_mask']
    """

    if fields is None and tags is None:
        raise ValueError("Please provide at least one parameter of `fields` or `tags`.")

    if fields is not None and fields.batch_size[0] != len(keys):
        raise ValueError(
            f"`keys` with length {len(keys)} does not match the `fields` TensorDict with "
            f"batch_size {fields.batch_size[0]}"
        )

    tq_client = _maybe_create_tq_client()

    # 1. translate user-specified key to BatchMeta
    batch_meta = await tq_client.async_kv_retrieve_meta(
        keys=keys,
        partition_id=partition_id,
        create=True,
        include_field_schema=False,
    )

    if batch_meta.size != len(keys):
        raise RuntimeError(
            f"Retrieved BatchMeta size {batch_meta.size} does not match with input `keys` size {len(keys)}!"
        )

    # 2. register the user-specified tags to BatchMeta
    if tags is not None:
        if len(tags) != len(keys):
            raise ValueError(f"keys with length {len(keys)} does not match length of tags {len(tags)}")
        batch_meta.update_custom_meta(tags)

    # 3. put data
    if fields is not None:
        # After put, batch_meta.field_names will include the new fields written by user
        # A fields-only write must not republish an older snapshot of the tags.
        batch_meta = await tq_client._async_put(
            fields, batch_meta, data_parser=data_parser, publish_custom_meta=tags is not None
        )
    else:
        # Directly update custom_meta (tags) to controller
        await tq_client.async_set_custom_meta(batch_meta)

    fields_to_return = batch_meta.field_names

    return KVBatchMeta(
        keys=keys,
        tags=batch_meta.custom_meta,
        partition_id=partition_id,
        fields=fields_to_return,
        extra_info=batch_meta.extra_info,
    )


async def async_kv_batch_get_by_meta(meta: KVBatchMeta, select_fields: list[str] | str | None = None) -> TensorDict:
    """Asynchronously get data from TransferQueue using KVBatchMeta.

    This is a convenience method for retrieving data using KVBatchMeta returned
    from a previous put operation. It extracts the keys and partition_id from
    the metadata to fetch the corresponding data.

    Args:
        meta: KVBatchMeta object returned from a previous put operation (e.g., async_kv_put,
              async_kv_batch_put). It contains keys, partition_id, and fields information.
        select_fields: Optional field(s) to retrieve, which overrides the fields
                       recorded in the given KVBatchMeta. If None, uses all fields
                       from meta.fields. Can be a single field name (str) or a list
                       of field names.

    Returns:
        TensorDict with the requested data

    Raises:
        ValueError: If keys or partition are not found
        ValueError: If empty fields exist in any key (sample)
        ValueError: If any field in select_fields doesn't exist in KVBatchMeta.fields

    Example:
        >>> import transfer_queue as tq
        >>> tq.init()
        >>> # First put some data
        >>> keys = ["sample_1", "sample_2", "sample_3"]
        >>> fields = TensorDict({
        ...     "input_ids": torch.randn(3, 10),
        ...     "attention_mask": torch.ones(3, 10),
        ... }, batch_size=3)
        >>> meta = await tq.async_kv_batch_put(keys=keys, partition_id="train", fields=fields)
        >>> # Then retrieve it using the returned metadata
        >>> data = await tq.async_kv_batch_get_by_meta(meta)
    """
    if meta.partition_id is None:
        raise ValueError("Must provide partition_id in the input KVBatchMeta.")

    fields_to_fetch: list[str] | None
    if select_fields is not None:
        if isinstance(select_fields, str):
            fields_to_fetch = [select_fields]
        else:
            fields_to_fetch = select_fields

        assert fields_to_fetch is not None
        if meta.fields is None or any(f not in meta.fields for f in fields_to_fetch):
            raise ValueError(
                f"Some fields assigned in select_fields not found in the metadata. "
                f"Assigned: {fields_to_fetch}; Fields in KVBatchMeta: {meta.fields}."
            )
    else:
        fields_to_fetch = meta.fields
    return await async_kv_batch_get(keys=meta.keys, partition_id=meta.partition_id, select_fields=fields_to_fetch)


async def async_kv_batch_get(
    keys: list[str] | str, partition_id: str, select_fields: list[str] | str | None = None
) -> TensorDict:
    """Asynchronously get data from TransferQueue using user-specified keys.

    This is a convenience method for retrieving data using keys instead of indexes.

    Args:
        keys: Single key or list of keys to retrieve
        partition_id: Partition containing the keys
        select_fields: Optional field(s) to retrieve. If None, retrieves all fields

    Returns:
        TensorDict with the requested data

    Raises:
        ValueError: If keys or partition are not found
        ValueError: If empty fields exist in any key (sample)

    Example:
        >>> import transfer_queue as tq
        >>> tq.init()
        >>> # Get single key with all fields
        >>> data = await tq.async_kv_batch_get(keys="sample_1", partition_id="train")
        >>> # Get multiple keys with specific fields
        >>> data = await tq.async_kv_batch_get(
        ...     keys=["sample_1", "sample_2"],
        ...     partition_id="train",
        ...     select_fields="input_ids"
        ... )
    """
    tq_client = _maybe_create_tq_client()

    fields_to_fetch = [select_fields] if isinstance(select_fields, str) else select_fields
    batch_meta = await tq_client.async_kv_retrieve_meta(
        keys=keys, partition_id=partition_id, create=False, data_fields=fields_to_fetch, include_custom_meta=False
    )

    if batch_meta.size == 0:
        raise ValueError("keys or partition were not found!")

    # Older controllers ignore the projection parameters in the request.
    if fields_to_fetch is not None and set(batch_meta.field_names) - set(fields_to_fetch):
        batch_meta = batch_meta.select_fields(fields_to_fetch)

    if not batch_meta.is_ready:
        raise ValueError("Some fields are not ready in all the requested keys!")

    data = await tq_client.async_get_data(batch_meta)
    return data


async def async_kv_list(partition_id: str | None = None) -> dict[str, dict[str, Any]]:
    """Asynchronously list all keys and their metadata in one or all partitions.

    Args:
        partition_id: The specific partition_id to query.
            If None (default), returns keys from all partitions.

    Returns:
        A nested dictionary mapping partition IDs to their keys and metadata.

        Structure:
        {
            "partition_id": {
                "key_name": {
                    "tag1": <value>,
                    ... (other metadata)
                },
                ...,
            },
            ...
        }


    Example:
        >>> import transfer_queue as tq
        >>> tq.init()
        >>> # Case 1: Retrieve a specific partition
        >>> partitions = await tq.async_kv_list(partition_id="train")
        >>> print(f"Keys: {list(partitions['train'].keys())}")
        >>> print(f"Tags: {list(partitions['train'].values())}")
        >>> # Case 2: Retrieve all partitions
        >>> all_partitions = await tq.async_kv_list()
        >>> for pid, keys in all_partitions.items():
        >>>     print(f"Partition: {pid}, Key count: {len(keys)}")
    """
    tq_client = _maybe_create_tq_client()

    partition_info = await tq_client.async_kv_list(partition_id)

    return partition_info


async def async_kv_clear(keys: list[str] | str, partition_id: str) -> None:
    """Asynchronously clear key-value pairs from TransferQueue.

    This removes the specified keys and their associated data from both
    the controller and storage units.

    Args:
        keys: Single key or list of keys to clear
        partition_id: Partition containing the keys

    Example:
        >>> import transfer_queue as tq
        >>> tq.init()
        >>> # Clear single key
        >>> await tq.async_kv_clear(keys="sample_1", partition_id="train")
        >>> # Clear multiple keys
        >>> await tq.async_kv_clear(keys=["sample_1", "sample_2"], partition_id="train")
    """

    if isinstance(keys, str):
        keys = [keys]

    tq_client = _maybe_create_tq_client()
    batch_meta = await tq_client.async_kv_retrieve_meta(keys=keys, partition_id=partition_id, create=False)

    if batch_meta.size > 0:
        await tq_client.async_clear_samples(batch_meta)


# ==================== Low-Level Native API ====================
# For low-level API support, please refer to transfer_queue/client.py for details.
def get_client():
    """Get a TransferQueueClient for using low-level API"""
    assert _TQ_CLIENT is not None, "Please initialize the TransferQueue first by calling `tq.init()`!"
    return _TQ_CLIENT


# ==================== Checkpoint API ====================

_METADATA_FILE = "metadata.json"
_CONTROLLER_FILE = "controller_state.pkl"
_CHECKPOINT_LOCK = Lock()
_PENDING_CHECKPOINT: Future | None = None
_PENDING_CHECKPOINT_DIR: Path | None = None


class CheckpointStorageError(RuntimeError):
    """A known payload I/O failure or deadline; settlement is queried separately."""

    def __init__(self, message, *, failures=(), timed_out=False):
        super().__init__(message)
        self.failures = tuple(failures)
        self.timed_out = timed_out
        self.checkpoint_future = None


class CheckpointStorageTimeoutError(CheckpointStorageError, TimeoutError):
    """A storage deadline does not attest that its writers have settled."""


def checkpoint_status(checkpoint_dir: str | Path, future: Future) -> dict:
    """Observe owned writers without publishing or clearing a failed save.

    A completed Future alone is not settlement: a timeout or one failed unit
    can precede other unit writers. Unknown transport/writer errors still raise.
    """
    with _CHECKPOINT_LOCK:
        path = Path(checkpoint_dir)
        if future is not _PENDING_CHECKPOINT or path != _PENDING_CHECKPOINT_DIR:
            raise ValueError("checkpoint status does not identify the owned pending save")
        if not future.done():
            return {"settled": False}
        error = future.exception()
        if error is None:
            return {"settled": True}
        if not isinstance(error, CheckpointStorageError):
            raise error
        result = cast(
            dict,
            _maybe_create_tq_client().save_storage_checkpoint(
                str(path.parent / f"{path.name}.tmp"), phase="status", publish_manifest=False
            ),
        )
        if any(not item["storage_io"] for item in result["failures"]):
            raise RuntimeError(f"Non-storage checkpoint failure: {result['failures']}") from error
        return {"settled": result["done"]}


def retire_checkpoint(checkpoint_dir: str | Path, future: Future) -> None:
    """Release failed-save bookkeeping after the caller settles the whole cut.

    This never deletes files. The enclosing checkpoint owner must disarm its
    other writers and remove only its uncommitted partial cut before retiring.
    """
    global _PENDING_CHECKPOINT, _PENDING_CHECKPOINT_DIR
    if not checkpoint_status(checkpoint_dir, future)["settled"]:
        raise RuntimeError("cannot retire unsettled checkpoint writers")
    with _CHECKPOINT_LOCK:
        if future is not _PENDING_CHECKPOINT or Path(checkpoint_dir) != _PENDING_CHECKPOINT_DIR:
            raise ValueError("checkpoint retirement does not identify the owned pending save")
        _PENDING_CHECKPOINT = None
        _PENDING_CHECKPOINT_DIR = None


def _commit_checkpoint(tmp_dir: Path, checkpoint_dir: Path, old_dir: Path, metadata: dict) -> None:
    """Publish only after payload files and their directory entries are durable."""
    with open(tmp_dir / _METADATA_FILE, "w") as stream:
        json.dump(metadata, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    for directory in [*sorted((p for p in tmp_dir.rglob("*") if p.is_dir()), reverse=True), tmp_dir]:
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    if checkpoint_dir.exists():
        if old_dir.exists():
            shutil.rmtree(old_dir)
        checkpoint_dir.rename(old_dir)
    tmp_dir.rename(checkpoint_dir)
    descriptor = os.open(checkpoint_dir.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if old_dir.exists():
        shutil.rmtree(old_dir)
    logger.info(f"Checkpoint saved to {checkpoint_dir}")


def save_checkpoint(
    checkpoint_dir: str | Path,
    *,
    include_storage: bool = True,
    metadata: dict[str, Any] | None = None,
    async_storage: bool = False,
    max_snapshot_bytes: int = 0,
    timeout_s: float = 900.0,
) -> Future | None:
    """Save controller and payload state on a shared filesystem.

    By default this blocks until publication. With ``async_storage=True``,
    SimpleStorage captures detached field maps before returning a Future. The
    caller must join that Future before publishing its enclosing checkpoint.
    Existing-row mutations must be quiesced through capture; fresh append-only
    producers may continue. Controller state, including sampler state, stays at
    the synchronous capture frontier. Unknown payload ownership or a retention
    estimate above ``max_snapshot_bytes`` falls back to a synchronous write.
    The positive byte budget covers retained payload stores and map estimates,
    not a process RSS limit or pickle's transient memory. Only one save can be
    pending; a failed save prevents another cut until its enclosing owner has
    settled every writer and explicitly retired the failed cut.
    Other storage backends retain their synchronous behavior. ``include_storage``
    can disable external-backend dumps; in-memory SimpleStorage always saves its
    payloads. ``metadata`` is copied into the existing metadata.json user section.
    """
    global _PENDING_CHECKPOINT, _PENDING_CHECKPOINT_DIR
    if _TQ_CONTROLLER is None:
        raise RuntimeError("TransferQueue is not initialized. Call tq.init() first.")
    if async_storage:
        if type(max_snapshot_bytes) is not int or max_snapshot_bytes <= 0:
            raise ValueError("max_snapshot_bytes must be a positive integer for async storage")
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be positive and finite")

    with _CHECKPOINT_LOCK:
        if _PENDING_CHECKPOINT is not None:
            _PENDING_CHECKPOINT.result(timeout=timeout_s)
        _PENDING_CHECKPOINT = None
        _PENDING_CHECKPOINT_DIR = None
        client = _maybe_create_tq_client()
        detached = async_storage and isinstance(client.storage_manager, AsyncSimpleStorageManager)
        if not include_storage and isinstance(client.storage_manager, AsyncSimpleStorageManager):
            logger.warning("SimpleStorage is in-memory; forcing include_storage=True.")
            include_storage = True
        checkpoint_dir = Path(checkpoint_dir)
        tmp_dir = checkpoint_dir.parent / f"{checkpoint_dir.name}.tmp"
        old_dir = checkpoint_dir.parent / f"{checkpoint_dir.name}.old"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True)
        writers_started = False
        try:
            controller_path = str(tmp_dir / _CONTROLLER_FILE)
            if detached:
                frontier = cast(dict, client.save_controller_checkpoint(controller_path, frontier=True))
            else:
                client.save_controller_checkpoint(controller_path)
            prepared = None
            if include_storage:
                try:
                    if detached:
                        # A failed fanout may already have started a subset of writers.
                        writers_started = True
                        _PENDING_CHECKPOINT_DIR = checkpoint_dir
                        prepared = client.save_storage_checkpoint(
                            str(tmp_dir),
                            phase="prepare",
                            indexes=frontier["indexes"],
                            max_snapshot_bytes=max_snapshot_bytes,
                        )
                    else:
                        if isinstance(client.storage_manager, AsyncSimpleStorageManager):
                            writers_started = True
                            _PENDING_CHECKPOINT_DIR = checkpoint_dir
                        result = client.save_storage_checkpoint(str(tmp_dir))
                        if result is not None and result["failures"]:
                            if any(not item["storage_io"] for item in result["failures"]):
                                raise RuntimeError(f"Non-storage checkpoint failure: {result['failures']}")
                            raise CheckpointStorageError(
                                f"TransferQueue payload I/O failed: {result['failures']}", failures=result["failures"]
                            )
                except NotImplementedError:
                    logger.warning("Storage backend does not support checkpoint; storage will not be saved.")
                    include_storage = False
            meta_content = {"storage_saved": include_storage, "user_metadata": copy.deepcopy(metadata or {})}
            if prepared is not None:
                # Keep only the bounded primitive receipt, not arbitrary replies.
                # Failure here must not hide any checkpoint I/O failure below.
                try:
                    receipt = {
                        "mode": "async" if prepared["prepared"] else "synchronous_fallback",
                        "max_snapshot_bytes": max_snapshot_bytes,
                        "storage_units": prepared["storage_units"],
                    }
                    meta_content["storage_checkpoint"] = json.loads(json.dumps(receipt, allow_nan=False))
                except Exception:
                    pass

            def finish():
                if detached:
                    deadline = time.monotonic() + timeout_s
                    while True:
                        status = client.save_storage_checkpoint(str(tmp_dir), phase="status")
                        if status["failures"]:
                            if any(not item["storage_io"] for item in status["failures"]):
                                raise RuntimeError(f"Non-storage checkpoint failure: {status['failures']}")
                            raise CheckpointStorageError(
                                f"TransferQueue payload I/O failed: {status['failures']}", failures=status["failures"]
                            )
                        if status["done"]:
                            break
                        if time.monotonic() >= deadline:
                            raise CheckpointStorageTimeoutError(
                                "TransferQueue storage checkpoint write timed out", timed_out=True
                            )
                        time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
                _commit_checkpoint(tmp_dir, checkpoint_dir, old_dir, meta_content)

            if prepared is not None and prepared["prepared"]:
                future: Future[None] = Future()
                future.set_running_or_notify_cancel()
                _PENDING_CHECKPOINT = future

                def complete():
                    try:
                        finish()
                    except BaseException as error:
                        if isinstance(error, CheckpointStorageError):
                            error.checkpoint_future = future
                        future.set_exception(error)
                    else:
                        future.set_result(None)

                Thread(target=complete, name="TransferQueueCheckpoint", daemon=True).start()
                return future
            finish()
            _PENDING_CHECKPOINT_DIR = None
            return None
        except BaseException as error:
            if writers_started:
                # A filesystem thread cannot be cancelled safely. Keep staging and
                # reject a second save rather than deleting files still being written.
                failed: Future[None] = Future()
                failed.set_exception(error)
                _PENDING_CHECKPOINT = failed
                if isinstance(error, CheckpointStorageError):
                    error.checkpoint_future = failed
                    if async_storage:
                        return failed
            elif tmp_dir.exists():
                shutil.rmtree(tmp_dir)
            if old_dir.exists() and not checkpoint_dir.exists():
                old_dir.rename(checkpoint_dir)
            raise


def load_checkpoint(
    checkpoint_dir: str | Path,
) -> None:
    """Restore TransferQueue system state from a checkpoint.

    .. note::
        **Multi-node limitation**: checkpoint_dir must reside on a shared network
        filesystem (e.g. NFS, GPFS, Lustre) accessible from all nodes.
        Single-node deployments have no such requirement.

    Args:
        checkpoint_dir: Path to the checkpoint directory.

    Raises:
        FileNotFoundError: Checkpoint directory or required files do not exist.
        RuntimeError: TransferQueue is not initialized, or restore fails.
    """
    if _TQ_CONTROLLER is None:
        raise RuntimeError("TransferQueue is not initialized. Call tq.init() first.")

    if _PENDING_CHECKPOINT is not None:
        _PENDING_CHECKPOINT.result(timeout=0)

    checkpoint_dir = Path(checkpoint_dir)
    old_dir = checkpoint_dir.parent / f"{checkpoint_dir.name}.old"

    if not checkpoint_dir.exists() and old_dir.exists():
        # A prior save_checkpoint crashed between renaming checkpoint_dir aside
        # and installing the new version; roll back to the pre-crash checkpoint.
        logger.warning(f"Found interrupted checkpoint replacement; restoring {old_dir} to {checkpoint_dir}")
        old_dir.rename(checkpoint_dir)

    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    metadata_path = checkpoint_dir / _METADATA_FILE
    if not metadata_path.exists():
        raise FileNotFoundError(f"{_METADATA_FILE} not found in {checkpoint_dir}")

    with open(metadata_path) as f:
        meta = json.load(f)

    client = _maybe_create_tq_client()

    controller_path = checkpoint_dir / _CONTROLLER_FILE
    if not controller_path.exists():
        raise FileNotFoundError(f"{_CONTROLLER_FILE} not found in {checkpoint_dir}")

    if meta.get("storage_saved"):
        client.load_storage_checkpoint(str(checkpoint_dir))

    client.load_controller_checkpoint(str(controller_path))

    logger.info(f"Checkpoint loaded from {checkpoint_dir}")
