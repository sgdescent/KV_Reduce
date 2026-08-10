import json
import os
import re
from typing import Any, Dict, List, Sequence, Tuple

import torch

from kv_utils import get_head_dim, get_num_kv_heads


FULL_PRECISION_BITS = 16
PER_TOKEN_AXIS = "per_token"
PER_CHANNEL_AXIS = "per_channel"


def dtype_bits(dtype_name: str) -> int:
    normalized = dtype_name.lower()
    if normalized in {"bf16", "bfloat16", "fp16", "float16", "half"}:
        return 16
    if normalized in {"fp32", "float32"}:
        return 32
    raise ValueError(f"Unsupported dtype for KV estimate: {dtype_name}")


def parse_csv_ints(value: str) -> List[int]:
    return [int(item.strip()) for item in re.split(r"[,;]", value) if item.strip()]


def quantize_dequantize_per_vector_symmetric(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Fake-quantize each [batch, head, token] KV vector and return dequantized values."""
    if bits >= FULL_PRECISION_BITS:
        return x
    if bits < 2:
        raise ValueError("Symmetric KV quantization needs at least 2 bits.")

    qmax = float((1 << (bits - 1)) - 1)
    scale = x.float().abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / qmax
    q = torch.round(x.float() / scale).clamp(-qmax, qmax)
    return (q * scale).to(dtype=x.dtype)


def _validate_grouping(group_size: int, residual_length: int) -> None:
    if group_size <= 0:
        raise ValueError("group_size must be positive.")
    if residual_length < 0:
        raise ValueError("residual_length must be non-negative.")


def per_channel_quantized_prefix_length(
    seq_len: int,
    *,
    group_size: int,
    residual_length: int,
) -> int:
    """Return the grouped prefix length, leaving a high-precision recent window."""
    _validate_grouping(group_size, residual_length)
    eligible = max(0, int(seq_len) - int(residual_length))
    return (eligible // int(group_size)) * int(group_size)


def quantize_dequantize_per_channel_grouped_symmetric(
    x: torch.Tensor,
    bits: int,
    *,
    group_size: int = 32,
) -> torch.Tensor:
    """Fake-quantize K per channel over complete token groups.

    Input is [..., token, channel]. Each channel receives one scale per token
    group, matching the key-axis choice used by KIVI-style quantization.
    """
    if bits >= FULL_PRECISION_BITS:
        return x
    if bits < 2:
        raise ValueError("Symmetric KV quantization needs at least 2 bits.")
    _validate_grouping(group_size, 0)
    token_count = int(x.shape[-2])
    if token_count == 0 or token_count % group_size != 0:
        raise ValueError(
            f"Per-channel input has {token_count} tokens; expected a multiple of group_size={group_size}."
        )

    qmax = float((1 << (bits - 1)) - 1)
    grouped = x.float().reshape(*x.shape[:-2], token_count // group_size, group_size, x.shape[-1])
    scale = grouped.abs().amax(dim=-2, keepdim=True).clamp_min(1e-8) / qmax
    q = torch.round(grouped / scale).clamp(-qmax, qmax)
    return (q * scale).reshape_as(x).to(dtype=x.dtype)


def quantize_dequantize_per_channel_grouped_affine(
    x: torch.Tensor,
    bits: int,
    *,
    group_size: int = 32,
) -> torch.Tensor:
    """Fake-quantize grouped keys with one affine range per channel."""
    if bits >= FULL_PRECISION_BITS:
        return x
    if bits < 2:
        raise ValueError("Affine KV quantization needs at least 2 bits.")
    _validate_grouping(group_size, 0)
    token_count = int(x.shape[-2])
    if token_count == 0 or token_count % group_size != 0:
        raise ValueError(
            f"Per-channel input has {token_count} tokens; expected a multiple of group_size={group_size}."
        )

    qmax = float((1 << bits) - 1)
    grouped = x.float().reshape(*x.shape[:-2], token_count // group_size, group_size, x.shape[-1])
    minimum = grouped.amin(dim=-2, keepdim=True)
    maximum = grouped.amax(dim=-2, keepdim=True)
    scale = (maximum - minimum).clamp_min(1e-8) / qmax
    q = torch.round((grouped - minimum) / scale).clamp(0.0, qmax)
    return (q * scale + minimum).reshape_as(x).to(dtype=x.dtype)


def quantize_key_cache_kivi_style(
    x: torch.Tensor,
    bits: int,
    *,
    group_size: int = 32,
    residual_length: int = 128,
    previous_seq_len: int = 0,
) -> torch.Tensor:
    """Quantize only newly eligible grouped K prefixes and preserve a BF16 tail."""
    if bits >= FULL_PRECISION_BITS:
        return x
    seq_len = int(x.shape[-2])
    if previous_seq_len < 0 or previous_seq_len > seq_len:
        raise ValueError(f"previous_seq_len={previous_seq_len} is invalid for seq_len={seq_len}.")
    previous_end = per_channel_quantized_prefix_length(
        previous_seq_len,
        group_size=group_size,
        residual_length=residual_length,
    )
    current_end = per_channel_quantized_prefix_length(
        seq_len,
        group_size=group_size,
        residual_length=residual_length,
    )
    if current_end <= previous_end:
        return x
    out = x.clone()
    out[..., previous_end:current_end, :] = quantize_dequantize_per_channel_grouped_affine(
        x[..., previous_end:current_end, :],
        bits,
        group_size=group_size,
    )
    return out


def uniform_bit_lists(num_layers: int, k_bits: int, v_bits: int) -> Tuple[List[int], List[int]]:
    return [int(k_bits)] * num_layers, [int(v_bits)] * num_layers


def _coerce_bits_list(values: Sequence[Any], num_layers: int, name: str) -> List[int]:
    if len(values) != num_layers:
        raise ValueError(f"{name} has {len(values)} entries, expected {num_layers}.")
    return [int(value) for value in values]


def load_bit_allocation(path: str, num_layers: int) -> Tuple[List[int], List[int], Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "k_bits" in data and "v_bits" in data:
        k_bits = _coerce_bits_list(data["k_bits"], num_layers, "k_bits")
        v_bits = _coerce_bits_list(data["v_bits"], num_layers, "v_bits")
        return k_bits, v_bits, data

    k_bits, v_bits = uniform_bit_lists(num_layers, FULL_PRECISION_BITS, FULL_PRECISION_BITS)
    for layer_cfg in data.get("layers", []):
        layer = int(layer_cfg["layer"])
        if layer < 0 or layer >= num_layers:
            raise ValueError(f"Layer {layer} in allocation is out of range for {num_layers} layers.")
        if "k_bits" in layer_cfg:
            k_bits[layer] = int(layer_cfg["k_bits"])
        if "v_bits" in layer_cfg:
            v_bits[layer] = int(layer_cfg["v_bits"])
    return k_bits, v_bits, data


def parse_quant_config_spec(spec: str, num_layers: int) -> Tuple[str, List[int], List[int], Dict[str, Any]]:
    """Parse compact config names like none, 8, k8v4, k=8:v=4, or allocation:path."""
    raw_spec = spec.strip()
    normalized = raw_spec.lower()
    metadata: Dict[str, Any] = {"spec": raw_spec}

    if normalized in {"none", "native", "bf16", "fp16", "full", "16"}:
        return "none", *uniform_bit_lists(num_layers, FULL_PRECISION_BITS, FULL_PRECISION_BITS), metadata

    if normalized.startswith("allocation:"):
        path = raw_spec.split(":", 1)[1]
        k_bits, v_bits, allocation = load_bit_allocation(path, num_layers)
        name = allocation.get("name") or os.path.splitext(os.path.basename(path))[0]
        metadata["allocation_path"] = path
        metadata["allocation"] = allocation
        return str(name), k_bits, v_bits, metadata

    if os.path.exists(raw_spec):
        k_bits, v_bits, allocation = load_bit_allocation(raw_spec, num_layers)
        name = allocation.get("name") or os.path.splitext(os.path.basename(raw_spec))[0]
        metadata["allocation_path"] = raw_spec
        metadata["allocation"] = allocation
        return str(name), k_bits, v_bits, metadata

    if normalized.startswith("int"):
        normalized = normalized[3:]
    if normalized.isdigit():
        bits = int(normalized)
        return f"k{bits}v{bits}", *uniform_bit_lists(num_layers, bits, bits), metadata

    match = re.fullmatch(r"k(\d+)v(\d+)", normalized)
    if match:
        k_bits = int(match.group(1))
        v_bits = int(match.group(2))
        return f"k{k_bits}v{v_bits}", *uniform_bit_lists(num_layers, k_bits, v_bits), metadata

    cleaned = normalized.replace("=", "").replace(":", "").replace("_", "")
    match = re.fullmatch(r"k(\d+),?v(\d+)", cleaned)
    if match:
        k_bits = int(match.group(1))
        v_bits = int(match.group(2))
        return f"k{k_bits}v{v_bits}", *uniform_bit_lists(num_layers, k_bits, v_bits), metadata

    raise ValueError(
        f"Unsupported quant config {raw_spec!r}. Use none, 8, k8v4, or allocation:/path/to/allocation.json."
    )


def parse_quant_config_specs(specs: str, num_layers: int) -> List[Tuple[str, List[int], List[int], Dict[str, Any]]]:
    configs = []
    # Semicolons are useful when passing the list through SLURM --export, whose
    # own syntax reserves commas as environment-variable separators.
    for item in re.split(r"[,;]", specs):
        item = item.strip()
        if not item:
            continue
        configs.append(parse_quant_config_spec(item, num_layers))
    if not configs:
        raise ValueError("At least one quantization config is required.")
    return configs


def quantize_legacy_cache(
    legacy_cache: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
    k_bits_by_layer: Sequence[int],
    v_bits_by_layer: Sequence[int],
    *,
    key_quant_axis: str = PER_TOKEN_AXIS,
    key_group_size: int = 32,
    key_residual_length: int = 128,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
    if len(legacy_cache) != len(k_bits_by_layer) or len(legacy_cache) != len(v_bits_by_layer):
        raise ValueError("Cache layer count and bit allocation length must match.")

    quantized = []
    for layer_idx, (k, v) in enumerate(legacy_cache):
        if key_quant_axis == PER_TOKEN_AXIS:
            quantized_key = quantize_dequantize_per_vector_symmetric(
                k, int(k_bits_by_layer[layer_idx])
            )
        elif key_quant_axis == PER_CHANNEL_AXIS:
            quantized_key = quantize_key_cache_kivi_style(
                k,
                int(k_bits_by_layer[layer_idx]),
                group_size=key_group_size,
                residual_length=key_residual_length,
            )
        else:
            raise ValueError(f"Unsupported key_quant_axis: {key_quant_axis!r}")
        quantized.append(
            (
                quantized_key,
                quantize_dequantize_per_vector_symmetric(v, int(v_bits_by_layer[layer_idx])),
            )
        )
    return tuple(quantized)


def estimate_model_kv_cache_bytes(
    *,
    config,
    seq_len: int,
    dtype_name: str,
    k_bits_by_layer: Sequence[int],
    v_bits_by_layer: Sequence[int],
    scale_bits: int = 16,
    batch_size: int = 1,
    key_quant_axis: str = PER_TOKEN_AXIS,
    key_group_size: int = 32,
    key_residual_length: int = 128,
) -> Dict[str, float]:
    num_layers = int(config.num_hidden_layers)
    if len(k_bits_by_layer) != num_layers or len(v_bits_by_layer) != num_layers:
        raise ValueError(f"Expected {num_layers} bit entries for model config.")

    num_kv_heads = get_num_kv_heads(config)
    head_dim = get_head_dim(config)
    vector_values = float(batch_size * num_kv_heads * seq_len * head_dim)
    per_token_scales = float(batch_size * num_kv_heads * seq_len)
    full_bits = dtype_bits(dtype_name)

    if key_quant_axis not in {PER_TOKEN_AXIS, PER_CHANNEL_AXIS}:
        raise ValueError(f"Unsupported key_quant_axis: {key_quant_axis!r}")
    key_quantized_tokens = (
        per_channel_quantized_prefix_length(
            seq_len,
            group_size=key_group_size,
            residual_length=key_residual_length,
        )
        if key_quant_axis == PER_CHANNEL_AXIS
        else seq_len
    )
    key_residual_tokens = seq_len - key_quantized_tokens
    per_channel_key_scales = (
        float(batch_size * num_kv_heads * (key_quantized_tokens // key_group_size) * head_dim)
        if key_quant_axis == PER_CHANNEL_AXIS
        else 0.0
    )

    native_bytes = 0.0
    quantized_bytes = 0.0
    for layer_idx in range(num_layers):
        k_bits = int(k_bits_by_layer[layer_idx])
        v_bits = int(v_bits_by_layer[layer_idx])
        native_bytes += 2.0 * vector_values * full_bits / 8.0

        if k_bits >= full_bits:
            quantized_bytes += vector_values * full_bits / 8.0
        elif key_quant_axis == PER_TOKEN_AXIS:
            quantized_bytes += vector_values * k_bits / 8.0
            quantized_bytes += per_token_scales * scale_bits / 8.0
        else:
            quantized_key_values = float(
                batch_size * num_kv_heads * key_quantized_tokens * head_dim
            )
            residual_key_values = float(
                batch_size * num_kv_heads * key_residual_tokens * head_dim
            )
            quantized_bytes += quantized_key_values * k_bits / 8.0
            quantized_bytes += residual_key_values * full_bits / 8.0
            # KIVI-style affine groups store both scale and zero-point/offset.
            quantized_bytes += 2.0 * per_channel_key_scales * scale_bits / 8.0

        if v_bits >= full_bits:
            quantized_bytes += vector_values * full_bits / 8.0
        else:
            quantized_bytes += vector_values * v_bits / 8.0
            quantized_bytes += per_token_scales * scale_bits / 8.0

    return {
        "native_cache_bytes": native_bytes,
        "quantized_cache_bytes": quantized_bytes,
        "cache_bytes_saved": native_bytes - quantized_bytes,
        "cache_saved_fraction": (native_bytes - quantized_bytes) / native_bytes if native_bytes > 0 else 0.0,
        "native_cache_mib": native_bytes / (1024.0**2),
        "quantized_cache_mib": quantized_bytes / (1024.0**2),
        "cache_mib_saved": (native_bytes - quantized_bytes) / (1024.0**2),
        "key_quantized_prefix_tokens": float(key_quantized_tokens),
        "key_residual_tokens": float(key_residual_tokens),
    }


def bit_allocation_stats(k_bits_by_layer: Sequence[int], v_bits_by_layer: Sequence[int]) -> Dict[str, float]:
    k_values = [float(x) for x in k_bits_by_layer]
    v_values = [float(x) for x in v_bits_by_layer]
    all_values = k_values + v_values
    return {
        "k_bits_min": min(k_values),
        "k_bits_max": max(k_values),
        "k_bits_mean": sum(k_values) / len(k_values),
        "v_bits_min": min(v_values),
        "v_bits_max": max(v_values),
        "v_bits_mean": sum(v_values) / len(v_values),
        "all_bits_mean": sum(all_values) / len(all_values),
    }
