"""Triton decode attention over bit-packed KIVI-style KV cache tensors."""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from packed_kv_cache import PackedKVComponent


SUPPORTED_BITS = {4, 8}


@triton.jit
def _load_unsigned(payload, value_indices, mask, BITS: tl.constexpr):
    if BITS == 8:
        return tl.load(payload + value_indices, mask=mask, other=0.0).to(tl.float32)
    byte_indices = value_indices // 2
    shifts = (value_indices % 2) * 4
    packed = tl.load(payload + byte_indices, mask=mask, other=0).to(tl.int32)
    return ((packed >> shifts) & 0xF).to(tl.float32)


@triton.jit
def _packed_kv_decode_kernel(
    query,
    key_payload,
    key_scale,
    key_offset,
    key_residual,
    value_payload,
    value_scale,
    value_offset,
    output,
    seq_len,
    prefix_len,
    residual_len,
    num_groups,
    q_heads: tl.constexpr,
    kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    group_size: tl.constexpr,
    K_BITS: tl.constexpr,
    V_BITS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    program = tl.program_id(0)
    batch_idx = program // q_heads
    query_head = program % q_heads
    kv_head = query_head // (q_heads // kv_heads)

    offsets_d = tl.arange(0, BLOCK_D)
    dim_mask = offsets_d < head_dim
    query_base = (batch_idx * q_heads + query_head) * head_dim
    q = tl.load(query + query_base + offsets_d, mask=dim_mask, other=0.0).to(tl.float32)

    running_max = -float("inf")
    running_sum = 0.0
    accumulator = tl.zeros((BLOCK_D,), dtype=tl.float32)
    score_scale = 1.0 / tl.sqrt(float(head_dim))

    for start_n in tl.range(0, seq_len, BLOCK_N):
        offsets_n = start_n + tl.arange(0, BLOCK_N)
        token_mask = offsets_n < seq_len
        matrix_mask = token_mask[:, None] & dim_mask[None, :]

        key_value_base = (batch_idx * kv_heads + kv_head) * seq_len * head_dim
        key_value_indices = key_value_base + offsets_n[:, None] * head_dim + offsets_d[None, :]
        prefix_mask = offsets_n < prefix_len
        quantized_key = _load_unsigned(
            key_payload,
            key_value_indices,
            matrix_mask & prefix_mask[:, None],
            BITS=K_BITS,
        )
        key_groups = offsets_n // group_size
        key_meta_base = (batch_idx * kv_heads + kv_head) * num_groups * head_dim
        key_meta_indices = key_meta_base + key_groups[:, None] * head_dim + offsets_d[None, :]
        key_scales = tl.load(
            key_scale + key_meta_indices,
            mask=matrix_mask & prefix_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        key_offsets = tl.load(
            key_offset + key_meta_indices,
            mask=matrix_mask & prefix_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        dequantized_key = quantized_key * key_scales + key_offsets

        residual_mask = token_mask & (offsets_n >= prefix_len)
        residual_base = (batch_idx * kv_heads + kv_head) * residual_len * head_dim
        residual_indices = residual_base + (offsets_n[:, None] - prefix_len) * head_dim + offsets_d[None, :]
        residual_key_values = tl.load(
            key_residual + residual_indices,
            mask=matrix_mask & residual_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        keys = tl.where(prefix_mask[:, None], dequantized_key, residual_key_values)

        scores = tl.sum(keys * q[None, :], axis=1) * score_scale
        scores = tl.where(token_mask, scores, -float("inf"))
        block_max = tl.max(scores, axis=0)
        new_max = tl.maximum(running_max, block_max)
        old_scale = tl.exp(running_max - new_max)
        probabilities = tl.exp(scores - new_max)

        value_value_base = (batch_idx * kv_heads + kv_head) * seq_len * head_dim
        value_indices = value_value_base + offsets_n[:, None] * head_dim + offsets_d[None, :]
        quantized_value = _load_unsigned(
            value_payload,
            value_indices,
            matrix_mask,
            BITS=V_BITS,
        )
        value_meta_base = (batch_idx * kv_heads + kv_head) * seq_len
        value_meta_indices = value_meta_base + offsets_n
        value_scales = tl.load(value_scale + value_meta_indices, mask=token_mask, other=0.0).to(tl.float32)
        value_offsets = tl.load(value_offset + value_meta_indices, mask=token_mask, other=0.0).to(tl.float32)
        values = quantized_value * value_scales[:, None] + value_offsets[:, None]

        accumulator = accumulator * old_scale + tl.sum(probabilities[:, None] * values, axis=0)
        running_sum = running_sum * old_scale + tl.sum(probabilities, axis=0)
        running_max = new_max

    output_base = (batch_idx * q_heads + query_head) * head_dim
    tl.store(output + output_base + offsets_d, accumulator / running_sum, mask=dim_mask)


def packed_kv_decode_attention(
    query: torch.Tensor,
    packed_keys: PackedKVComponent,
    packed_values: PackedKVComponent,
    *,
    block_tokens: int = 32,
) -> torch.Tensor:
    """Run one-token GQA attention without materializing the packed cache."""
    if not query.is_cuda:
        raise ValueError("Triton packed attention requires a CUDA query tensor.")
    if query.ndim != 4 or query.shape[-2] != 1:
        raise ValueError("query must have shape [batch, query_heads, 1, head_dim].")
    if packed_keys.layout != "per_channel_grouped_affine":
        raise ValueError("keys must use the grouped per-channel affine layout.")
    if packed_values.layout != "per_token_affine":
        raise ValueError("values must use the per-token affine layout.")
    if packed_keys.bits not in SUPPORTED_BITS or packed_values.bits not in SUPPORTED_BITS:
        raise ValueError("The fused kernel currently supports only 4-bit and 8-bit K/V payloads.")
    if packed_keys.shape != packed_values.shape:
        raise ValueError("Packed key and value shapes must match.")
    if tuple(query.shape[:1]) != tuple(packed_keys.shape[:1]):
        raise ValueError("Query and cache batch dimensions must match.")
    if query.shape[-1] != packed_keys.shape[-1]:
        raise ValueError("Query and cache head dimensions must match.")
    if query.shape[1] % packed_keys.shape[1] != 0:
        raise ValueError("Query heads must be divisible by KV heads.")
    if packed_keys.prefix_length <= 0:
        raise ValueError("The fused kernel requires a non-empty quantized key prefix.")
    if block_tokens not in {16, 32, 64}:
        raise ValueError("block_tokens must be one of 16, 32, or 64.")

    batch, kv_heads, seq_len, head_dim = packed_keys.shape
    query_heads = int(query.shape[1])
    residual_len = int(seq_len - packed_keys.prefix_length)
    num_groups = int(packed_keys.prefix_length // packed_keys.group_size)
    block_dim = triton.next_power_of_2(head_dim)
    output = torch.empty_like(query)
    grid = (int(batch * query_heads),)
    _packed_kv_decode_kernel[grid](
        query,
        packed_keys.payload,
        packed_keys.scale,
        packed_keys.offset,
        packed_keys.residual,
        packed_values.payload,
        packed_values.scale,
        packed_values.offset,
        output,
        int(seq_len),
        int(packed_keys.prefix_length),
        residual_len,
        num_groups,
        q_heads=query_heads,
        kv_heads=int(kv_heads),
        head_dim=int(head_dim),
        group_size=int(packed_keys.group_size),
        K_BITS=int(packed_keys.bits),
        V_BITS=int(packed_values.bits),
        BLOCK_N=int(block_tokens),
        BLOCK_D=int(block_dim),
        num_warps=4 if block_dim <= 128 else 8,
    )
    return output


def theoretical_attention_flops(query: torch.Tensor, seq_len: int) -> int:
    """Approximate dot-product plus value-accumulation FLOPs for one decode query."""
    batch, query_heads, _, head_dim = query.shape
    return int(4 * batch * query_heads * seq_len * head_dim)

