# Copyright 2026 Zyphra Technologies Inc.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Compact physical row storage that retains each field's public dtype and shape."""

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


class NibbleRoutes:
    """Route ids in 0..15 stored two per byte; restores the packed uint8 routes."""

    __slots__ = ("data", "shape")

    def __init__(self, routes: PackedRoutes):
        flat = routes.data.flatten()
        if flat.numel() % 2:
            flat = torch.cat([flat, flat.new_zeros(1)])
        self.data, self.shape = flat[0::2] | (flat[1::2] << 4), tuple(routes.data.shape)

    def unpack(self) -> PackedRoutes:
        flat = torch.stack([self.data & 15, self.data >> 4], dim=1).flatten()
        return PackedRoutes(flat[: _numel(self.shape)].reshape(self.shape))

    def __reduce__(self):
        return self.unpack().__reduce__()


class NarrowInts:
    """An int64 row stored in the narrowest integer dtype that holds its values."""

    __slots__ = ("data",)

    def __init__(self, data):
        self.data = data

    def restore(self):
        return self.data.to(torch.int64)

    def __reduce__(self):
        # Ordinary Torch readers restore int64.
        return torch._utils._rebuild_device_tensor_from_numpy, (self.data.numpy(), torch.int64, "cpu", False)


class Arange:
    """A one-dimensional int64 row equal to ``arange(start, stop)``, such as text position ids."""

    __slots__ = ("start", "stop")

    def __init__(self, start, stop):
        self.start, self.stop = start, stop

    def restore(self):
        return torch.arange(self.start, self.stop)

    def __reduce__(self):
        return torch.arange, (self.start, self.stop)


def _numel(shape):
    count = 1
    for size in shape:
        count *= size
    return count


_NARROW_DTYPES = (torch.uint8, torch.int8, torch.int16, torch.int32)
# Smaller rows (scalars, short lists) keep their tensor: compaction would save little.
_COMPACT_MIN_NUMEL = 512


def compact_row(value):
    """Store a row value compactly when its bytes allow; everything else is returned unchanged."""
    if isinstance(value, PackedRoutes):
        data = value.data
        if not value.offset and data.numel() >= _COMPACT_MIN_NUMEL and int(data.max()) <= 15:
            return NibbleRoutes(value)
        return value
    if (
        type(value) is not torch.Tensor
        or value.dtype != torch.int64
        or value.device.type != "cpu"
        or value.layout != torch.strided
        or value.is_nested
        or not value.is_contiguous()
        or value.requires_grad
        or value.numel() < _COMPACT_MIN_NUMEL
        or value.__dict__
        or any(name is not None for name in getattr(value, "names", ()))
    ):
        return value
    if value.dim() == 1:
        start = int(value[0])
        if torch.equal(value, torch.arange(start, start + value.numel())):
            return Arange(start, start + value.numel())
    low, high = int(value.min()), int(value.max())
    for dtype in _NARROW_DTYPES:
        info = torch.iinfo(dtype)
        if info.min <= low and high <= info.max:
            return NarrowInts(value.to(dtype))
    return value


def restore_row(value, *, packed_routes=False):
    """The public value of a stored row; routes stay packed for transport when asked."""
    if isinstance(value, NibbleRoutes):
        value = value.unpack()
    if isinstance(value, NarrowInts | Arange) or (isinstance(value, PackedRoutes) and not packed_routes):
        return value.restore()
    return value


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
        if isinstance(value, PackedRoutes | NibbleRoutes | NarrowInts):
            pending.append(value.data)
        elif isinstance(value, Arange):
            continue
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
