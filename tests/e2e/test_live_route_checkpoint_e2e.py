# Copyright 2026 Zyphra Technologies Inc.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Public KV route schemas and sampled fields survive a real save/restore."""

import hashlib
import tempfile

import pytest
import ray
import torch
from omegaconf import OmegaConf
from tensordict import NonTensorStack, TensorDict

import transfer_queue as tq


@pytest.fixture
def system():
    assert not ray.is_initialized()
    directory = tempfile.TemporaryDirectory(prefix="tq-rc-")
    ray.init(
        address="local",
        num_cpus=3,
        num_gpus=0,
        include_dashboard=False,
        object_store_memory=128 * 1024**2,
        _node_ip_address="127.0.0.1",
        _temp_dir=directory.name,
    )
    config = OmegaConf.create(
        {
            "controller": {"polling_mode": True},
            "metrics": {"enabled": False},
            "backend": {
                "storage_backend": "SimpleStorage",
                "SimpleStorage": {
                    "total_storage_size": 32,
                    "num_data_storage_units": 2,
                },
            },
        }
    )
    try:
        tq.init(config)
        yield
    finally:
        tq.close()
        ray.shutdown()
        directory.cleanup()


def tensor_hash(value):
    digest = hashlib.sha256()
    for row in value.unbind(0):
        digest.update(str((row.dtype, tuple(row.shape))).encode())
        digest.update(row.contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def test_public_kv_metadata_and_async_checkpoint_roundtrip(system, tmp_path):
    keys, partition = ["row-a", "row-b"], "live-route"
    routes = torch.arange(640, dtype=torch.int16).reshape(2, 8, 40, 1) % 16
    routes[0, 0, 0, 0] = -1
    fields = TensorDict(
        {
            "routed_experts": routes,
            "input_ids": torch.arange(16, dtype=torch.int64).reshape(2, 8),
            "position_ids": torch.arange(8, dtype=torch.int64).expand(2, 8).clone(),
            "response_mask": torch.tensor([[0, 0, 1, 1, 1, 1, 0, 0]] * 2),
            "rollout_log_probs": torch.linspace(-3, -0.1, 16).reshape(2, 8),
            "metrics": NonTensorStack({"reward": 1.0}, {"reward": 0.0}),
        },
        batch_size=[2],
    )
    expected = {name: tensor_hash(value) for name, value in fields.items() if isinstance(value, torch.Tensor)}
    tags = [{"status": "completed", "session_id": index, "global_steps": 9} for index in range(2)]
    tq.kv_batch_put(keys=keys, partition_id=partition, fields=fields, tags=tags)
    meta = tq.get_client().kv_retrieve_meta(keys, partition, create=False)
    assert meta.field_schema["routed_experts"]["dtype"] == torch.int16
    before = tq.kv_batch_get(keys=keys, partition_id=partition)
    assert {name: tensor_hash(before[name]) for name in expected} == expected
    path = tmp_path / "checkpoint"
    future = tq.save_checkpoint(path, async_storage=True, max_snapshot_bytes=16 * 1024**2, timeout_s=30)
    assert future is not None
    future.result(timeout=30)
    assert (path / "metadata.json").is_file()
    tq.kv_clear(keys=keys, partition_id=partition)
    tq.load_checkpoint(path)
    after = tq.kv_batch_get(keys=keys, partition_id=partition)
    restored_meta = tq.get_client().kv_retrieve_meta(keys, partition, create=False)
    assert restored_meta.field_schema["routed_experts"] == meta.field_schema["routed_experts"]
    assert {name: tensor_hash(after[name]) for name in expected} == expected
    assert tq.kv_list(partition_id=partition)[partition] == dict(zip(keys, tags, strict=True))
