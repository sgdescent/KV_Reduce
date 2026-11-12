"""Actual bit-packed KV tensors for storage and unfused systems baselines."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Tuple

import torch


FULL_PRECISION_BITS = 16


def _quantized_prefix_length(seq_len: int, group_size: int, residual_length: int) -> int:
    if group_size <= 0 or residual_length < 0:
        raise ValueError("group_size must be positive and residual_length non-negative.")
    eligible = max(0, int(seq_len) - int(residual_length))
    return (eligible // int(group_size)) * int(group_size)


def packed_num_bytes(num_values: int, bits: int) -> int:
    if bits < 1 or bits > 8:
        raise ValueError("Packed values require between 1 and 8 bits.")
    return ceil(int(num_values) * int(bits) / 8)


def pack_unsigned(values: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack unsigned integer values into a flat uint8 payload."""
    if bits < 1 or bits > 8:
        raise ValueError("bits must be in [1, 8].")
    flat = values.reshape(-1).to(torch.uint8)
    if flat.numel() == 0:
        return torch.empty(0, dtype=torch.uint8, device=values.device)
    maximum = int(flat.max().item())
    if maximum >= (1 << bits):
        raise ValueError(f"Value {maximum} cannot be represented with {bits} bits.")
    if bits == 8:
        return flat.contiguous()
    if bits in {2, 4}:
        values_per_byte = 8 // bits
        padding = (-flat.numel()) % values_per_byte
        if padding:
            flat = torch.cat((flat, torch.zeros(padding, dtype=torch.uint8, device=flat.device)))
        grouped = flat.reshape(-1, values_per_byte).to(torch.int32)
        shifts = torch.arange(values_per_byte, device=flat.device, dtype=torch.int32) * bits
        return torch.sum(grouped << shifts, dim=-1).to(torch.uint8)

    flat_i32 = flat.to(torch.int32)
    indices = torch.arange(flat.numel(), device=flat.device, dtype=torch.int64)
    bit_positions = indices * bits
    byte_indices = torch.div(bit_positions, 8, rounding_mode="floor")
    offsets = torch.remainder(bit_positions, 8).to(torch.int32)
    packed = torch.zeros(packed_num_bytes(flat.numel(), bits), dtype=torch.int32, device=flat.device)
    packed.scatter_add_(0, byte_indices, (flat_i32 << offsets) & 0xFF)
    crosses = offsets + bits > 8
    if bool(crosses.any()):
        packed.scatter_add_(
            0,
            byte_indices[crosses] + 1,
            flat_i32[crosses] >> (8 - offsets[crosses]),
        )
    return packed.to(torch.uint8)


def unpack_unsigned(payload: torch.Tensor, bits: int, num_values: int) -> torch.Tensor:
    """Unpack a flat uint8 payload into unsigned uint8 values."""
    if bits < 1 or bits > 8:
        raise ValueError("bits must be in [1, 8].")
    if payload.dtype != torch.uint8:
        raise ValueError("payload must have dtype torch.uint8.")
    expected = packed_num_bytes(num_values, bits)
    if payload.numel() != expected:
        raise ValueError(f"Expected {expected} packed bytes, received {payload.numel()}.")
    if num_values == 0:
        return torch.empty(0, dtype=torch.uint8, device=payload.device)
    if bits == 8:
        return payload[:num_values].contiguous()
    if bits in {2, 4}:
        values_per_byte = 8 // bits
        shifts = torch.arange(values_per_byte, device=payload.device, dtype=torch.int32) * bits
        unpacked = (payload.to(torch.int32).unsqueeze(-1) >> shifts) & ((1 << bits) - 1)
        return unpacked.reshape(-1)[:num_values].to(torch.uint8)

    indices = torch.arange(num_values, device=payload.device, dtype=torch.int64)
    bit_positions = indices * bits
    byte_indices = torch.div(bit_positions, 8, rounding_mode="floor")
    offsets = torch.remainder(bit_positions, 8).to(torch.int32)
    payload_i32 = payload.to(torch.int32)
    raw = payload_i32[byte_indices] >> offsets
    crosses = offsets + bits > 8
    if bool(crosses.any()):
        raw[crosses] |= payload_i32[byte_indices[crosses] + 1] << (8 - offsets[crosses])
    return (raw & ((1 << bits) - 1)).to(torch.uint8)


def tensor_bytes(tensor: torch.Tensor | None) -> int:
    return 0 if tensor is None else tensor.numel() * tensor.element_size()


@dataclass
class PackedKVComponent:
    payload: torch.Tensor
    scale: torch.Tensor | None
    offset: torch.Tensor | None
    residual: torch.Tensor | None
    shape: Tuple[int, ...]
    bits: int
    layout: str
    output_dtype: torch.dtype
    group_size: int = 0
    prefix_length: int = 0

    @property
    def payload_bytes(self) -> int:
        return tensor_bytes(self.payload) + tensor_bytes(self.residual)

    @property
    def metadata_bytes(self) -> int:
        return tensor_bytes(self.scale) + tensor_bytes(self.offset)

    @property
    def storage_bytes(self) -> int:
        return self.payload_bytes + self.metadata_bytes

    def unpack(self) -> torch.Tensor:
        if self.layout == "native":
            return self.payload.reshape(self.shape)
        if self.layout == "per_token_affine":
            quantized = unpack_unsigned(self.payload, self.bits, _numel(self.shape)).reshape(self.shape)
            return (quantized.float() * self.scale.float() + self.offset.float()).to(self.output_dtype)
        if self.layout == "per_channel_grouped_affine":
            prefix_shape = (*self.shape[:-2], self.prefix_length, self.shape[-1])
            grouped_shape = (
                *self.shape[:-2],
                self.prefix_length // self.group_size,
                self.group_size,
                self.shape[-1],
            )
            quantized = unpack_unsigned(
                self.payload,
                self.bits,
                _numel(prefix_shape),
            ).reshape(grouped_shape)
            prefix = (quantized.float() * self.scale.float() + self.offset.float()).reshape(prefix_shape)
            if self.residual is not None and self.residual.shape[-2] > 0:
                prefix = torch.cat((prefix.to(self.output_dtype), self.residual), dim=-2)
            return prefix.to(self.output_dtype)
        raise ValueError(f"Unsupported packed layout: {self.layout}")


def _numel(shape: Tuple[int, ...]) -> int:
    result = 1
    for dimension in shape:
        result *= int(dimension)
    return result


def pack_per_token_affine(
    values: torch.Tensor,
    bits: int,
    *,
    metadata_dtype: torch.dtype = torch.bfloat16,
) -> PackedKVComponent:
    if bits >= FULL_PRECISION_BITS:
        return PackedKVComponent(
            payload=values.contiguous(),
            scale=None,
            offset=None,
            residual=None,
            shape=tuple(values.shape),
            bits=FULL_PRECISION_BITS,
            layout="native",
            output_dtype=values.dtype,
        )
    if bits < 2 or bits > 8:
        raise ValueError("Affine KV quantization supports 2 through 8 bits.")
    values_f32 = values.float()
    minimum = values_f32.amin(dim=-1, keepdim=True)
    maximum = values_f32.amax(dim=-1, keepdim=True)
    scale = (maximum - minimum).clamp_min(1e-8) / float((1 << bits) - 1)
    quantized = torch.round((values_f32 - minimum) / scale).clamp(0, (1 << bits) - 1).to(torch.uint8)
    return PackedKVComponent(
        payload=pack_unsigned(quantized, bits),
        scale=scale.to(metadata_dtype),
        offset=minimum.to(metadata_dtype),
        residual=None,
        shape=tuple(values.shape),
        bits=bits,
        layout="per_token_affine",
        output_dtype=values.dtype,
    )


def pack_per_channel_grouped_affine(
    keys: torch.Tensor,
    bits: int,
    *,
    group_size: int = 32,
    residual_length: int = 128,
    metadata_dtype: torch.dtype = torch.bfloat16,
) -> PackedKVComponent:
    if bits >= FULL_PRECISION_BITS:
        return PackedKVComponent(
            payload=keys.contiguous(),
            scale=None,
            offset=None,
            residual=None,
            shape=tuple(keys.shape),
            bits=FULL_PRECISION_BITS,
            layout="native",
            output_dtype=keys.dtype,
        )
    if bits < 2 or bits > 8:
        raise ValueError("Affine KV quantization supports 2 through 8 bits.")
    prefix_length = _quantized_prefix_length(keys.shape[-2], group_size, residual_length)
    prefix = keys[..., :prefix_length, :]
    residual = keys[..., prefix_length:, :].contiguous()
    if prefix_length == 0:
        return PackedKVComponent(
            payload=torch.empty(0, dtype=torch.uint8, device=keys.device),
            scale=torch.empty(0, dtype=metadata_dtype, device=keys.device),
            offset=torch.empty(0, dtype=metadata_dtype, device=keys.device),
            residual=residual,
            shape=tuple(keys.shape),
            bits=bits,
            layout="per_channel_grouped_affine",
            output_dtype=keys.dtype,
            group_size=group_size,
            prefix_length=0,
        )
    grouped = prefix.float().reshape(
        *keys.shape[:-2],
        prefix_length // group_size,
        group_size,
        keys.shape[-1],
    )
    minimum = grouped.amin(dim=-2, keepdim=True)
    maximum = grouped.amax(dim=-2, keepdim=True)
    scale = (maximum - minimum).clamp_min(1e-8) / float((1 << bits) - 1)
    quantized = torch.round((grouped - minimum) / scale).clamp(0, (1 << bits) - 1).to(torch.uint8)
    return PackedKVComponent(
        payload=pack_unsigned(quantized, bits),
        scale=scale.to(metadata_dtype),
        offset=minimum.to(metadata_dtype),
        residual=residual,
        shape=tuple(keys.shape),
        bits=bits,
        layout="per_channel_grouped_affine",
        output_dtype=keys.dtype,
        group_size=group_size,
        prefix_length=prefix_length,
    )


def pack_kivi_kv(
    keys: torch.Tensor,
    values: torch.Tensor,
    *,
    k_bits: int,
    v_bits: int,
    group_size: int = 32,
    residual_length: int = 128,
    metadata_dtype: torch.dtype = torch.bfloat16,
) -> Tuple[PackedKVComponent, PackedKVComponent]:
    return (
        pack_per_channel_grouped_affine(
            keys,
            k_bits,
            group_size=group_size,
            residual_length=residual_length,
            metadata_dtype=metadata_dtype,
        ),
        pack_per_token_affine(values, v_bits, metadata_dtype=metadata_dtype),
    )
