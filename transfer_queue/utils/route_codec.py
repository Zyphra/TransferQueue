# Copyright 2026 Zyphra Technologies Inc.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Physical route bytes retain the public int16 field contract."""

import torch
from tensordict import TensorDictBase


class PackedRoutes:
    __slots__ = ("_data", "_offset")

    def __init__(self, data, offset=0):
        if type(data) is not torch.Tensor or data.dtype != torch.uint8 or data.device.type != "cpu":
            raise ValueError("Packed routes require a CPU uint8 tensor")
        if type(offset) is not int or offset not in (0, -1):
            raise ValueError("Packed route offset must be zero or the -1 sentinel shift")
        self._data, self._offset = data, offset

    @property
    def data(self):
        return self._data

    @property
    def offset(self):
        return self._offset

    def restore(self):
        value = self.data.to(torch.int16)
        return value.add_(self.offset) if self.offset else value

    def __reduce__(self):
        # Ordinary Torch readers restore the original logical dtype.
        if self.offset:
            return torch.add, (PackedRoutes(self.data), self.offset)
        return torch._utils._rebuild_device_tensor_from_numpy, (self.data.numpy(), torch.int16, "cpu", False)


def packable_route_ids(fields, *, reject_identity_aliases=False):
    rows = fields.get("routed_experts", {})
    routes = {id(value) for value in (rows.values() if isinstance(rows, dict) else rows) if type(value) is torch.Tensor}
    if not routes:
        return set()
    pending, seen, owners, counts, projections = [fields], set(), {}, {}, []
    while pending:
        value = pending.pop()
        if type(value) is torch.Tensor:
            counts[id(value)] = counts.get(id(value), 0) + 1
        if id(value) in seen:
            continue
        seen.add(id(value))
        if isinstance(value, PackedRoutes):
            pending.append(value.data)
        elif type(value) is torch.Tensor:
            if value.device.type != "cpu" or value.layout != torch.strided or value.is_nested:
                return set()
            storage = value.untyped_storage()
            identity = (storage.data_ptr(), storage.nbytes())
            owners.setdefault(identity, []).append(value)
        elif isinstance(value, TensorDictBase):
            # Keep projected containers alive while their identities are in seen.
            projections.append(value.to_dict())
            pending.append(projections[-1])
        elif type(value) is dict:
            pending.extend(value.keys())
            pending.extend(value.values())
        elif type(value) in (list, tuple, set, frozenset):
            pending.extend(value)
        elif type(value) not in (str, bytes, bytearray, int, float, bool, type(None)):
            return set()
    for aliases in owners.values():
        if len(aliases) <= 1:
            continue
        route_rows = all(
            id(value) in routes and value.dtype == torch.int16 and value.is_contiguous() for value in aliases
        )
        spans = sorted((value.storage_offset(), value.storage_offset() + value.numel()) for value in aliases)
        if not route_rows or any(left[1] > right[0] for left, right in zip(spans, spans[1:], strict=False)):
            routes.difference_update(id(value) for value in aliases)
    if reject_identity_aliases:
        routes.difference_update(identity for identity, count in counts.items() if count > 1)
    return routes


def pack_route_fields(fields):
    """Pack independent route cells; unsupported values retain their old path."""
    routes = fields.get("routed_experts")
    if routes is None:
        return fields
    rows = dict(routes) if isinstance(routes, dict) else dict(enumerate(routes))
    normalized = {**fields, "routed_experts": rows}
    eligible = packable_route_ids(normalized, reject_identity_aliases=True)
    for index, value in rows.items():
        if (
            id(value) not in eligible
            or type(value) is not torch.Tensor
            or value.dtype != torch.int16
            or not value.is_contiguous()
            or value.is_neg()
            or value.__dict__
            or any(name is not None for name in getattr(value, "names", ()))
        ):
            continue
        array = value.numpy()
        low, high = (int(array.min()), int(array.max())) if array.size else (0, 0)
        if low >= 0 and high <= 255:
            rows[index] = PackedRoutes(value.to(torch.uint8))
        elif low == -1 and high <= 254:
            rows[index] = PackedRoutes((value + 1).to(torch.uint8), offset=-1)
    return {**fields, "routed_experts": rows if isinstance(routes, dict) else list(rows.values())}
