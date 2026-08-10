#!/usr/bin/env python3
"""
Benchmark sensitivity-aware draft KV-cache quantization inside speculative decoding.

The target verifier remains unquantized, so draft KV compression does not change the
target distribution used for verification. BF16 batched verification can still choose
a different top-1 token than tokenwise BF16 decoding at numerical ties; both margins
are recorded explicitly. Only the draft model's cached K/V tensors are fake-quantized.

  draft KV bytes saved vs. speculative acceptance retained.
"""

import argparse
import atexit
import csv
import inspect
import json
import math
import os
import time
from typing import Any, Dict, List, Optional, Sequence

import torch
import transformers

from kv_cache_quantization import (
    PER_CHANNEL_AXIS,
    PER_TOKEN_AXIS,
    bit_allocation_stats,
    estimate_model_kv_cache_bytes,
    parse_quant_config_specs,
    quantize_dequantize_per_vector_symmetric,
    quantize_key_cache_kivi_style,
    quantize_legacy_cache,
    uniform_bit_lists,
)
from kv_utils import (
    as_legacy_cache,
    distribution_metrics,
    iter_token_blocks,
    legacy_to_cache,
    load_causal_lm,
    load_tokenizer,
    set_seed,
    tokenizer_compatibility_report,
    write_json,
)


def write_csv(rows: List[Dict[str, Any]], path: str) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_csv_items(value: Optional[str]) -> List[str]:
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def aggregate_rows(rows: List[Dict[str, Any]], exclude: Optional[Sequence[str]] = None) -> Dict[str, float]:
    exclude = set(exclude or [])
    out: Dict[str, float] = {}
    if not rows:
        return out
    keys: List[str] = []
    for row in rows:
        for key, value in row.items():
            if key in exclude or isinstance(value, str):
                continue
            if key not in keys:
                keys.append(key)
    for key in keys:
        values = [float(row[key]) for row in rows if key in row and not isinstance(row[key], str)]
        if values:
            out[key] = float(sum(values) / len(values))
    return out


def init_wandb(args: argparse.Namespace) -> Optional[Any]:
    if not args.wandb:
        return None
    try:
        import wandb
    except ImportError as e:
        raise ImportError("W&B logging requested but wandb is not installed.") from e

    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        entity=args.wandb_entity,
        group=args.wandb_group,
        config=vars(args),
    )

    def _finish_wandb() -> None:
        if run is not None:
            run.finish()

    atexit.register(_finish_wandb)
    return run


def cuda_devices(*devices: str) -> List[int]:
    if not torch.cuda.is_available():
        return []
    out: List[int] = []
    for device in devices:
        if not str(device).startswith("cuda"):
            continue
        index = torch.device(device).index
        out.append(0 if index is None else int(index))
    return sorted(set(out))


def sync_cuda(devices: Sequence[int]) -> None:
    for device in devices:
        torch.cuda.synchronize(device)


def reset_cuda_peak(devices: Sequence[int]) -> None:
    for device in devices:
        torch.cuda.reset_peak_memory_stats(device)


def cuda_memory_snapshot(devices: Sequence[int], prefix: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for device in devices:
        label = f"{prefix}/cuda:{device}"
        out[f"{label}/allocated_bytes"] = float(torch.cuda.memory_allocated(device))
        out[f"{label}/reserved_bytes"] = float(torch.cuda.memory_reserved(device))
        out[f"{label}/peak_allocated_bytes"] = float(torch.cuda.max_memory_allocated(device))
        out[f"{label}/peak_reserved_bytes"] = float(torch.cuda.max_memory_reserved(device))
        out[f"{label}/allocated_mib"] = out[f"{label}/allocated_bytes"] / (1024.0**2)
        out[f"{label}/reserved_mib"] = out[f"{label}/reserved_bytes"] / (1024.0**2)
        out[f"{label}/peak_allocated_mib"] = out[f"{label}/peak_allocated_bytes"] / (1024.0**2)
        out[f"{label}/peak_reserved_mib"] = out[f"{label}/peak_reserved_bytes"] / (1024.0**2)
    return out


def _ones_attention_mask(total_len: int, device: str) -> torch.Tensor:
    return torch.ones((1, total_len), dtype=torch.long, device=device)


def _cache_position(start: int, length: int, device: str) -> torch.Tensor:
    return torch.arange(start, start + length, dtype=torch.long, device=device)


def shared_token_logits(logits: torch.Tensor, shared_vocab_size: int) -> torch.Tensor:
    """Drop model-output padding so compatible tokenizers use identical support."""
    if shared_vocab_size <= 0:
        raise ValueError("shared_vocab_size must be positive.")
    if logits.shape[-1] < shared_vocab_size:
        raise ValueError(
            f"Model exposes {logits.shape[-1]} logits, fewer than the shared tokenizer vocabulary "
            f"of {shared_vocab_size}."
        )
    return logits[..., :shared_vocab_size]


def top1_logit_margin(logits: torch.Tensor) -> float:
    top_two = torch.topk(logits.float(), k=2, dim=-1).values
    return float((top_two[..., 0] - top_two[..., 1]).mean().item())


def quantize_cache_for_next_step(
    past_key_values,
    k_bits: Sequence[int],
    v_bits: Sequence[int],
    *,
    new_tokens: Optional[int] = None,
    key_quant_axis: str = PER_TOKEN_AXIS,
    key_group_size: int = 32,
    key_residual_length: int = 128,
):
    if all(int(bits) >= 16 for bits in k_bits) and all(int(bits) >= 16 for bits in v_bits):
        return past_key_values
    if new_tokens is not None and new_tokens <= 0:
        raise ValueError("new_tokens must be positive when provided.")
    if hasattr(past_key_values, "layers"):
        for layer_idx, layer in enumerate(past_key_values.layers):
            token_slice = slice(None) if new_tokens is None else slice(-new_tokens, None)
            value_slice = layer.values[..., token_slice, :]
            quantized_value = quantize_dequantize_per_vector_symmetric(value_slice, int(v_bits[layer_idx]))
            if key_quant_axis == PER_TOKEN_AXIS:
                key_slice = layer.keys[..., token_slice, :]
                quantized_key = quantize_dequantize_per_vector_symmetric(
                    key_slice, int(k_bits[layer_idx])
                )
                if new_tokens is None:
                    layer.keys = quantized_key
                else:
                    key_slice.copy_(quantized_key)
            elif key_quant_axis == PER_CHANNEL_AXIS:
                previous_seq_len = 0 if new_tokens is None else int(layer.keys.shape[-2]) - new_tokens
                layer.keys = quantize_key_cache_kivi_style(
                    layer.keys,
                    int(k_bits[layer_idx]),
                    group_size=key_group_size,
                    residual_length=key_residual_length,
                    previous_seq_len=previous_seq_len,
                )
            else:
                raise ValueError(f"Unsupported key_quant_axis: {key_quant_axis!r}")
            if new_tokens is None:
                layer.values = quantized_value
            else:
                value_slice.copy_(quantized_value)
        return past_key_values
    legacy = as_legacy_cache(past_key_values)
    quantized = quantize_legacy_cache(
        legacy,
        k_bits,
        v_bits,
        key_quant_axis=key_quant_axis,
        key_group_size=key_group_size,
        key_residual_length=key_residual_length,
    )
    return legacy_to_cache(quantized)


def crop_cache_to_length(past_key_values, length: int):
    """Crop a cache after speculative rollback without recomputing the prefix."""
    if length < 0:
        raise ValueError("Cache length must be non-negative.")
    if hasattr(past_key_values, "crop"):
        past_key_values.crop(length)
        return past_key_values
    legacy = as_legacy_cache(past_key_values)
    cropped = tuple(
        (
            key[..., :length, :].contiguous(),
            value[..., :length, :].contiguous(),
        )
        for key, value in legacy
    )
    return legacy_to_cache(cropped)


def last_token_logits_kwargs(model) -> Dict[str, int]:
    """Request only final-token logits when the model API supports it."""
    try:
        parameters = inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        return {}
    if "logits_to_keep" in parameters:
        return {"logits_to_keep": 1}
    if "num_logits_to_keep" in parameters:
        return {"num_logits_to_keep": 1}
    return {}


@torch.no_grad()
def cached_prefill(model, input_ids: torch.Tensor, device: str) -> Dict[str, Any]:
    input_on_device = input_ids.to(device)
    out = model(
        input_ids=input_on_device,
        use_cache=True,
        **last_token_logits_kwargs(model),
    )
    return {
        "logits": out.logits[:, -1, :],
        "cache": out.past_key_values,
        "cache_len": int(input_on_device.shape[1]),
    }


@torch.no_grad()
def cached_step(
    *,
    model,
    input_ids: torch.Tensor,
    cache,
    cache_len: int,
    device: str,
) -> Dict[str, Any]:
    input_on_device = input_ids.to(device)
    step_len = int(input_on_device.shape[1])
    out = model(
        input_ids=input_on_device,
        attention_mask=_ones_attention_mask(cache_len + step_len, device),
        past_key_values=cache,
        use_cache=True,
        cache_position=_cache_position(cache_len, step_len, device),
    )
    return {
        "logits": out.logits,
        "cache": out.past_key_values,
        "cache_len": cache_len + step_len,
    }


@torch.no_grad()
def greedy_target_generate_with_margins(
    *,
    big_model,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    big_device: str,
    shared_vocab_size: int,
) -> Dict[str, List[Any]]:
    state = cached_prefill(big_model, prompt_ids, big_device)
    logits = shared_token_logits(state["logits"], shared_vocab_size)
    cache = state["cache"]
    cache_len = int(state["cache_len"])
    generated: List[int] = []
    margins: List[float] = []
    for token_idx in range(max_new_tokens):
        margins.append(top1_logit_margin(logits))
        token = int(logits.argmax(dim=-1).item())
        generated.append(token)
        if token_idx + 1 >= max_new_tokens:
            break
        step = cached_step(
            model=big_model,
            input_ids=torch.tensor([[token]], dtype=prompt_ids.dtype),
            cache=cache,
            cache_len=cache_len,
            device=big_device,
        )
        logits = shared_token_logits(step["logits"][:, -1, :], shared_vocab_size)
        cache = step["cache"]
        cache_len = int(step["cache_len"])
    return {"tokens": generated, "top1_margins": margins}


@torch.no_grad()
def greedy_target_generate(
    *,
    big_model,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    big_device: str,
    shared_vocab_size: int,
) -> List[int]:
    return greedy_target_generate_with_margins(
        big_model=big_model,
        prompt_ids=prompt_ids,
        max_new_tokens=max_new_tokens,
        big_device=big_device,
        shared_vocab_size=shared_vocab_size,
    )["tokens"]


@torch.no_grad()
def draft_next_logits_from_cache(
    *,
    small_model,
    token: int,
    cache,
    cache_len: int,
    small_device: str,
    dtype: torch.dtype,
    k_bits: Sequence[int],
    v_bits: Sequence[int],
    key_quant_axis: str,
    key_group_size: int,
    key_residual_length: int,
) -> Dict[str, Any]:
    input_ids = torch.tensor([[token]], dtype=dtype, device=small_device)
    step = cached_step(
        model=small_model,
        input_ids=input_ids,
        cache=cache,
        cache_len=cache_len,
        device=small_device,
    )
    cache = quantize_cache_for_next_step(
        step["cache"],
        k_bits,
        v_bits,
        new_tokens=1,
        key_quant_axis=key_quant_axis,
        key_group_size=key_group_size,
        key_residual_length=key_residual_length,
    )
    return {"logits": step["logits"][:, -1, :], "cache": cache, "cache_len": int(step["cache_len"])}


@torch.no_grad()
def greedy_speculative_decode_cached_quantized(
    *,
    big_model,
    small_model,
    prompt_ids: torch.Tensor,
    draft_steps: int,
    max_new_tokens: int,
    big_device: str,
    small_device: str,
    topk: int,
    k_bits: Sequence[int],
    v_bits: Sequence[int],
    shared_vocab_size: int,
    key_quant_axis: str = PER_TOKEN_AXIS,
    key_group_size: int = 32,
    key_residual_length: int = 128,
) -> Dict[str, Any]:
    target_state = cached_prefill(big_model, prompt_ids, big_device)
    target_logits = shared_token_logits(target_state["logits"], shared_vocab_size)
    target_cache = target_state["cache"]
    target_cache_len = int(target_state["cache_len"])

    draft_state = cached_prefill(small_model, prompt_ids, small_device)
    draft_logits = shared_token_logits(draft_state["logits"], shared_vocab_size)
    draft_cache = quantize_cache_for_next_step(
        draft_state["cache"],
        k_bits,
        v_bits,
        key_quant_axis=key_quant_axis,
        key_group_size=key_group_size,
        key_residual_length=key_residual_length,
    )
    draft_cache_len = int(draft_state["cache_len"])

    if target_cache_len != draft_cache_len:
        raise ValueError(
            f"Target and draft prefills produced different cache lengths: {target_cache_len} vs {draft_cache_len}."
        )

    generated: List[int] = []
    proposed_tokens = 0
    accepted_tokens = 0
    target_calls = 1
    target_verify_calls = 0
    draft_decode_calls = 0
    draft_prefill_calls = 1
    draft_prefill_tokens = int(prompt_ids.shape[1])
    full_accept_rounds = 0
    num_rounds = 0
    round_metric_rows: List[Dict[str, float]] = []
    generation_sources: List[str] = []
    target_top1_margins: List[float] = []

    while len(generated) < max_new_tokens:
        num_rounds += 1
        round_prefix_len = target_cache_len
        proposal: List[int] = []

        max_round_steps = min(draft_steps, max_new_tokens - len(generated))
        for proposal_idx in range(max_round_steps):
            if proposal_idx == 0:
                metrics = distribution_metrics(
                    target_logits.to(draft_logits.device),
                    draft_logits,
                    topk=topk,
                )
                round_metric_rows.append(metrics)

            token = int(draft_logits.argmax(dim=-1).item())
            proposal.append(token)
            next_step = draft_next_logits_from_cache(
                small_model=small_model,
                token=token,
                cache=draft_cache,
                cache_len=draft_cache_len,
                small_device=small_device,
                dtype=prompt_ids.dtype,
                k_bits=k_bits,
                v_bits=v_bits,
                key_quant_axis=key_quant_axis,
                key_group_size=key_group_size,
                key_residual_length=key_residual_length,
            )
            draft_decode_calls += 1
            draft_logits = shared_token_logits(next_step["logits"], shared_vocab_size)
            draft_cache = next_step["cache"]
            draft_cache_len = int(next_step["cache_len"])

        proposed_tokens += len(proposal)
        verify_ids = torch.tensor([proposal], dtype=prompt_ids.dtype)
        verify = cached_step(
            model=big_model,
            input_ids=verify_ids,
            cache=target_cache,
            cache_len=target_cache_len,
            device=big_device,
        )
        target_calls += 1
        target_verify_calls += 1
        verify_logits = shared_token_logits(verify["logits"], shared_vocab_size)
        target_cache = verify["cache"]
        target_cache_len = int(verify["cache_len"])

        accepted_this_round = 0
        proposal_target_margins: List[float] = []
        for idx, token in enumerate(proposal):
            token_logits = target_logits if idx == 0 else verify_logits[:, idx - 1, :]
            proposal_target_margins.append(top1_logit_margin(token_logits))
            target_token = int(token_logits.argmax(dim=-1).item())
            if target_token != token:
                break
            accepted_this_round += 1

        accepted_tokens += accepted_this_round
        if accepted_this_round == len(proposal):
            full_accept_rounds += 1

        generated.extend(proposal[:accepted_this_round])
        generation_sources.extend(["accepted_proposal"] * accepted_this_round)
        target_top1_margins.extend(proposal_target_margins[:accepted_this_round])

        if len(generated) >= max_new_tokens:
            break

        correction_logits = target_logits if accepted_this_round == 0 else verify_logits[:, accepted_this_round - 1, :]
        correction_token = int(correction_logits.argmax(dim=-1).item())
        generated.append(correction_token)
        generation_sources.append(
            "verified_bonus" if accepted_this_round == len(proposal) else "target_correction"
        )
        target_top1_margins.append(top1_logit_margin(correction_logits))

        committed_len = round_prefix_len + accepted_this_round
        target_cache = crop_cache_to_length(target_cache, committed_len)
        draft_cache = crop_cache_to_length(draft_cache, committed_len)
        target_cache_len = committed_len
        draft_cache_len = committed_len

        target_commit = cached_step(
            model=big_model,
            input_ids=torch.tensor([[correction_token]], dtype=prompt_ids.dtype),
            cache=target_cache,
            cache_len=target_cache_len,
            device=big_device,
        )
        target_calls += 1
        target_logits = shared_token_logits(target_commit["logits"][:, -1, :], shared_vocab_size)
        target_cache = target_commit["cache"]
        target_cache_len = int(target_commit["cache_len"])

        draft_commit = draft_next_logits_from_cache(
            small_model=small_model,
            token=correction_token,
            cache=draft_cache,
            cache_len=draft_cache_len,
            small_device=small_device,
            dtype=prompt_ids.dtype,
            k_bits=k_bits,
            v_bits=v_bits,
            key_quant_axis=key_quant_axis,
            key_group_size=key_group_size,
            key_residual_length=key_residual_length,
        )
        draft_decode_calls += 1
        draft_logits = shared_token_logits(draft_commit["logits"], shared_vocab_size)
        draft_cache = draft_commit["cache"]
        draft_cache_len = int(draft_commit["cache_len"])

    round_metrics = aggregate_rows(round_metric_rows)
    return {
        "generated_tokens": generated[:max_new_tokens],
        "generation_sources": generation_sources[:max_new_tokens],
        "target_top1_margins": target_top1_margins[:max_new_tokens],
        "proposed_tokens": int(proposed_tokens),
        "accepted_tokens": int(accepted_tokens),
        "accept_rate": float(accepted_tokens / proposed_tokens) if proposed_tokens > 0 else 0.0,
        "accepted_per_round": float(accepted_tokens / num_rounds) if num_rounds > 0 else 0.0,
        "full_accept_round_fraction": float(full_accept_rounds / num_rounds) if num_rounds > 0 else 0.0,
        "target_calls": int(target_calls),
        "target_verify_calls": int(target_verify_calls),
        "draft_decode_calls": int(draft_decode_calls),
        "draft_prefill_calls": int(draft_prefill_calls),
        "draft_prefill_tokens": int(draft_prefill_tokens),
        "num_rounds": int(num_rounds),
        "round_metrics": round_metrics,
    }


def estimate_total_kv_memory(
    *,
    big_model,
    small_model,
    big_dtype: str,
    small_dtype: str,
    seq_len: int,
    k_bits: Sequence[int],
    v_bits: Sequence[int],
    scale_bits: int,
    key_quant_axis: str = PER_TOKEN_AXIS,
    key_group_size: int = 32,
    key_residual_length: int = 128,
) -> Dict[str, float]:
    big_layers = int(big_model.config.num_hidden_layers)
    target_k_bits, target_v_bits = uniform_bit_lists(big_layers, 16, 16)
    target = estimate_model_kv_cache_bytes(
        config=big_model.config,
        seq_len=seq_len,
        dtype_name=big_dtype,
        k_bits_by_layer=target_k_bits,
        v_bits_by_layer=target_v_bits,
        scale_bits=scale_bits,
    )
    draft_full_k_bits, draft_full_v_bits = uniform_bit_lists(int(small_model.config.num_hidden_layers), 16, 16)
    draft_native = estimate_model_kv_cache_bytes(
        config=small_model.config,
        seq_len=seq_len,
        dtype_name=small_dtype,
        k_bits_by_layer=draft_full_k_bits,
        v_bits_by_layer=draft_full_v_bits,
        scale_bits=scale_bits,
    )
    draft_quant = estimate_model_kv_cache_bytes(
        config=small_model.config,
        seq_len=seq_len,
        dtype_name=small_dtype,
        k_bits_by_layer=k_bits,
        v_bits_by_layer=v_bits,
        scale_bits=scale_bits,
        key_quant_axis=key_quant_axis,
        key_group_size=key_group_size,
        key_residual_length=key_residual_length,
    )

    native_total = target["native_cache_bytes"] + draft_native["native_cache_bytes"]
    quant_total = target["native_cache_bytes"] + draft_quant["quantized_cache_bytes"]
    return {
        "seq_len": float(seq_len),
        "target_cache_bytes": target["native_cache_bytes"],
        "native_draft_cache_bytes": draft_native["native_cache_bytes"],
        "quantized_draft_cache_bytes": draft_quant["quantized_cache_bytes"],
        "native_total_cache_bytes": native_total,
        "quantized_total_cache_bytes": quant_total,
        "draft_cache_saved_fraction": draft_quant["cache_saved_fraction"],
        "total_cache_saved_fraction": (native_total - quant_total) / native_total if native_total > 0 else 0.0,
        "target_cache_mib": target["native_cache_mib"],
        "native_draft_cache_mib": draft_native["native_cache_mib"],
        "quantized_draft_cache_mib": draft_quant["quantized_cache_mib"],
        "native_total_cache_mib": native_total / (1024.0**2),
        "quantized_total_cache_mib": quant_total / (1024.0**2),
        "total_cache_mib_saved": (native_total - quant_total) / (1024.0**2),
        "key_quantized_prefix_tokens": draft_quant["key_quantized_prefix_tokens"],
        "key_residual_tokens": draft_quant["key_residual_tokens"],
        **{f"allocation/{key}": value for key, value in bit_allocation_stats(k_bits, v_bits).items()},
    }


@torch.no_grad()
def generate_target_references(
    *,
    prompts: Sequence[torch.Tensor],
    big_model,
    max_new_tokens: int,
    big_device: str,
    shared_vocab_size: int,
) -> List[List[int]]:
    return [record["tokens"] for record in generate_target_reference_records(
        prompts=prompts,
        big_model=big_model,
        max_new_tokens=max_new_tokens,
        big_device=big_device,
        shared_vocab_size=shared_vocab_size,
    )]


@torch.no_grad()
def generate_target_reference_records(
    *,
    prompts: Sequence[torch.Tensor],
    big_model,
    max_new_tokens: int,
    big_device: str,
    shared_vocab_size: int,
) -> List[Dict[str, List[Any]]]:
    return [
        greedy_target_generate_with_margins(
            big_model=big_model,
            prompt_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            big_device=big_device,
            shared_vocab_size=shared_vocab_size,
        )
        for prompt_ids in prompts
    ]


@torch.no_grad()
def run_one_config(
    *,
    config_name: str,
    prompts: Sequence[torch.Tensor],
    big_model,
    small_model,
    draft_steps: int,
    max_new_tokens: int,
    big_device: str,
    small_device: str,
    topk: int,
    k_bits: Sequence[int],
    v_bits: Sequence[int],
    cuda_device_ids: Sequence[int],
    wandb_run: Optional[Any],
    wandb_prefix: str,
    wandb_step_offset: int,
    shared_vocab_size: int,
    key_quant_axis: str = PER_TOKEN_AXIS,
    key_group_size: int = 32,
    key_residual_length: int = 128,
    target_token_references: Optional[Sequence[Sequence[int]]] = None,
    target_margin_references: Optional[Sequence[Sequence[float]]] = None,
) -> Dict[str, Any]:
    if target_token_references is None:
        reference_records = generate_target_reference_records(
            prompts=prompts,
            big_model=big_model,
            max_new_tokens=max_new_tokens,
            big_device=big_device,
            shared_vocab_size=shared_vocab_size,
        )
        target_token_references = [record["tokens"] for record in reference_records]
        target_margin_references = [record["top1_margins"] for record in reference_records]
    if len(target_token_references) != len(prompts):
        raise ValueError(
            f"Expected {len(prompts)} target references, received {len(target_token_references)}."
        )
    if target_margin_references is not None and len(target_margin_references) != len(prompts):
        raise ValueError(
            f"Expected {len(prompts)} target margin references, received {len(target_margin_references)}."
        )

    rows: List[Dict[str, Any]] = []
    reset_cuda_peak(cuda_device_ids)
    sync_cuda(cuda_device_ids)
    start = time.perf_counter()

    for prompt_idx, (prompt_ids, target_tokens) in enumerate(zip(prompts, target_token_references)):
        sync_cuda(cuda_device_ids)
        prompt_start = time.perf_counter()
        result = greedy_speculative_decode_cached_quantized(
            big_model=big_model,
            small_model=small_model,
            prompt_ids=prompt_ids,
            draft_steps=draft_steps,
            max_new_tokens=max_new_tokens,
            big_device=big_device,
            small_device=small_device,
            topk=topk,
            k_bits=k_bits,
            v_bits=v_bits,
            shared_vocab_size=shared_vocab_size,
            key_quant_axis=key_quant_axis,
            key_group_size=key_group_size,
            key_residual_length=key_residual_length,
        )
        sync_cuda(cuda_device_ids)
        elapsed_s = time.perf_counter() - prompt_start
        generated_tokens = len(result["generated_tokens"])
        target_tokens_list = list(target_tokens)
        first_mismatch = next(
            (
                token_idx
                for token_idx, (generated_token, target_token) in enumerate(
                    zip(result["generated_tokens"], target_tokens_list)
                )
                if generated_token != target_token
            ),
            -1,
        )
        mismatch_source = (
            result["generation_sources"][first_mismatch]
            if 0 <= first_mismatch < len(result["generation_sources"])
            else ""
        )
        mismatch_verifier_margin = (
            result["target_top1_margins"][first_mismatch]
            if 0 <= first_mismatch < len(result["target_top1_margins"])
            else float("nan")
        )
        prompt_reference_margins = (
            list(target_margin_references[prompt_idx]) if target_margin_references is not None else []
        )
        mismatch_reference_margin = (
            float(prompt_reference_margins[first_mismatch])
            if 0 <= first_mismatch < len(prompt_reference_margins)
            else float("nan")
        )
        finite_margins = [
            margin
            for margin in (mismatch_verifier_margin, mismatch_reference_margin)
            if not math.isnan(margin)
        ]
        mismatch_min_margin = min(finite_margins) if finite_margins else float("nan")
        row = {
            "config": config_name,
            "prompt_idx": int(prompt_idx),
            "latency_s": float(elapsed_s),
            "generated_tokens": int(generated_tokens),
            "tokens_per_second": float(generated_tokens / elapsed_s) if elapsed_s > 0 else 0.0,
            "ms_per_generated_token": float(1000.0 * elapsed_s / generated_tokens) if generated_tokens > 0 else 0.0,
            "matches_target_greedy": float(result["generated_tokens"] == target_tokens_list),
            "first_target_mismatch": int(first_mismatch),
            "mismatch_source": mismatch_source,
            "mismatch_target_top1_margin": float(mismatch_verifier_margin),
            "mismatch_verifier_top1_margin": float(mismatch_verifier_margin),
            "mismatch_reference_top1_margin": float(mismatch_reference_margin),
            "mismatch_min_top1_margin": float(mismatch_min_margin),
            "generated_token_ids": json.dumps(result["generated_tokens"]),
            "target_token_ids": json.dumps(target_tokens_list),
            "generation_sources": json.dumps(result["generation_sources"]),
            "target_top1_margins": json.dumps(result["target_top1_margins"]),
            "target_reference_top1_margins": json.dumps(prompt_reference_margins),
            "accept_rate": float(result["accept_rate"]),
            "accepted_per_round": float(result["accepted_per_round"]),
            "full_accept_round_fraction": float(result["full_accept_round_fraction"]),
            "proposed_tokens": int(result["proposed_tokens"]),
            "accepted_tokens": int(result["accepted_tokens"]),
            "target_calls": int(result["target_calls"]),
            "target_verify_calls": int(result["target_verify_calls"]),
            "draft_decode_calls": int(result["draft_decode_calls"]),
            "draft_prefill_calls": int(result["draft_prefill_calls"]),
            "draft_prefill_tokens": int(result["draft_prefill_tokens"]),
            "num_rounds": int(result["num_rounds"]),
        }
        for key, value in result["round_metrics"].items():
            row[f"round_{key}"] = float(value)
        rows.append(row)

        if wandb_run is not None:
            wandb_run.log(
                {
                    f"{wandb_prefix}/{config_name}/{key}": value
                    for key, value in row.items()
                    if key not in {"config", "prompt_idx"}
                },
                step=wandb_step_offset + prompt_idx + 1,
            )

    sync_cuda(cuda_device_ids)
    total_s = time.perf_counter() - start
    memory = cuda_memory_snapshot(cuda_device_ids, f"memory/{config_name}")
    summary = aggregate_rows(rows, exclude=["prompt_idx"])
    total_generated = sum(int(row["generated_tokens"]) for row in rows)
    total_proposed = sum(int(row["proposed_tokens"]) for row in rows)
    total_accepted = sum(int(row["accepted_tokens"]) for row in rows)
    summary.update(
        {
            "num_prompts": float(len(rows)),
            "total_latency_s": float(total_s),
            "total_generated_tokens": float(total_generated),
            "overall_tokens_per_second": float(total_generated / total_s) if total_s > 0 else 0.0,
            "overall_ms_per_generated_token": float(1000.0 * total_s / total_generated) if total_generated > 0 else 0.0,
            "overall_accept_rate": float(total_accepted / total_proposed) if total_proposed > 0 else 0.0,
            **memory,
        }
    )

    if wandb_run is not None:
        for key, value in summary.items():
            wandb_run.summary[f"{wandb_prefix}/{config_name}/{key}"] = value

    return {"rows": rows, "summary": summary}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark draft KV quantization inside cached speculative decoding.")
    parser.add_argument("--big_model", type=str, default="Qwen/Qwen2.5-3B")
    parser.add_argument("--small_model", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--big_device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--small_device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--big_dtype", type=str, default="bf16")
    parser.add_argument("--small_dtype", type=str, default="bf16")
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="sdpa",
        choices=["eager", "sdpa", "flash_attention_2"],
        help="Attention backend. SDPA avoids eager attention's quadratic score-matrix allocation.",
    )
    parser.add_argument("--dataset_name", type=str, default="wikitext")
    parser.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--text_file", type=str, default=None)
    parser.add_argument("--text_column", type=str, default=None)
    parser.add_argument("--eval_split", type=str, default="validation")
    parser.add_argument("--eval_split_fallbacks", type=str, default="test,train")
    parser.add_argument("--stream_eval", action="store_true")
    parser.add_argument("--shuffle_eval", action="store_true")
    parser.add_argument("--prompt_len", type=int, default=1024)
    parser.add_argument("--num_prompts", type=int, default=100)
    parser.add_argument("--warmup_prompts", type=int, default=5)
    parser.add_argument("--draft_steps", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument(
        "--quant_configs",
        type=str,
        default="none,8,k8v4,k4v8,k4v4",
        help="Comma-separated configs: none, 8, 4, k8v4, k4v8, or allocation:path.json.",
    )
    parser.add_argument("--scale_bits", type=int, default=16)
    parser.add_argument(
        "--key_quant_axis",
        type=str,
        default=PER_TOKEN_AXIS,
        choices=[PER_TOKEN_AXIS, PER_CHANNEL_AXIS],
        help="Quantize keys per token vector or per channel over token groups.",
    )
    parser.add_argument("--key_group_size", type=int, default=32)
    parser.add_argument(
        "--key_residual_length",
        type=int,
        default=128,
        help="Recent key tokens kept at full precision for per-channel quantization.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow_incompatible_tokenizers", action="store_true")
    parser.add_argument("--out_dir", type=str, default="outputs/spec_kv_quant_benchmark")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="kv-reduce")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    wandb_run = init_wandb(args)

    print("Loading models and tokenizers...")
    big_tokenizer = load_tokenizer(args.big_model)
    small_tokenizer = load_tokenizer(args.small_model)
    compatibility = tokenizer_compatibility_report(big_tokenizer, small_tokenizer)
    tokenizers_compatible = bool(
        compatibility["all_probe_encodings_match"] and compatibility["same_vocab_size"]
    )
    if (not tokenizers_compatible) and (not args.allow_incompatible_tokenizers):
        raise ValueError("Tokenizers appear incompatible. Use --allow_incompatible_tokenizers to override.")
    shared_vocab_size = min(int(big_tokenizer.vocab_size), int(small_tokenizer.vocab_size))

    big_model = load_causal_lm(
        args.big_model,
        device=args.big_device,
        dtype_name=args.big_dtype,
        attn_implementation=args.attn_implementation,
    )
    small_model = load_causal_lm(
        args.small_model,
        device=args.small_device,
        dtype_name=args.small_dtype,
        attn_implementation=args.attn_implementation,
    )
    quant_configs = parse_quant_config_specs(args.quant_configs, int(small_model.config.num_hidden_layers))
    print("Quant configs:", [name for name, _, _, _ in quant_configs])

    prompt_iter = iter_token_blocks(
        tokenizer=big_tokenizer,
        seq_len=args.prompt_len,
        max_blocks=args.num_prompts + args.warmup_prompts,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.eval_split,
        text_file=args.text_file,
        text_column=args.text_column,
        shuffle=args.shuffle_eval,
        seed=args.seed,
        streaming=args.stream_eval,
        split_fallbacks=parse_csv_items(args.eval_split_fallbacks),
    )
    all_prompts = [block.unsqueeze(0) for block in prompt_iter]
    warmup_prompts = all_prompts[: args.warmup_prompts]
    benchmark_prompts = all_prompts[args.warmup_prompts :]
    if not benchmark_prompts:
        raise ValueError("No benchmark prompts were loaded.")

    cuda_device_ids = cuda_devices(args.big_device, args.small_device)
    print("Generating cached target references once per prompt...")
    warmup_reference_records = generate_target_reference_records(
        prompts=warmup_prompts,
        big_model=big_model,
        max_new_tokens=args.max_new_tokens,
        big_device=args.big_device,
        shared_vocab_size=shared_vocab_size,
    )
    benchmark_reference_records = generate_target_reference_records(
        prompts=benchmark_prompts,
        big_model=big_model,
        max_new_tokens=args.max_new_tokens,
        big_device=args.big_device,
        shared_vocab_size=shared_vocab_size,
    )
    warmup_target_references = [record["tokens"] for record in warmup_reference_records]
    warmup_target_margins = [record["top1_margins"] for record in warmup_reference_records]
    benchmark_target_references = [record["tokens"] for record in benchmark_reference_records]
    benchmark_target_margins = [record["top1_margins"] for record in benchmark_reference_records]
    if warmup_prompts:
        print(f"Running {len(warmup_prompts)} warmup prompts for each config...")
        for name, k_bits, v_bits, _ in quant_configs:
            run_one_config(
                config_name=name,
                prompts=warmup_prompts,
                big_model=big_model,
                small_model=small_model,
                draft_steps=args.draft_steps,
                max_new_tokens=args.max_new_tokens,
                big_device=args.big_device,
                small_device=args.small_device,
                topk=args.topk,
                k_bits=k_bits,
                v_bits=v_bits,
                cuda_device_ids=cuda_device_ids,
                wandb_run=None,
                wandb_prefix="warmup",
                wandb_step_offset=0,
                shared_vocab_size=shared_vocab_size,
                key_quant_axis=args.key_quant_axis,
                key_group_size=args.key_group_size,
                key_residual_length=args.key_residual_length,
                target_token_references=warmup_target_references,
                target_margin_references=warmup_target_margins,
            )

    all_rows: List[Dict[str, Any]] = []
    summaries: Dict[str, Dict[str, float]] = {}
    memory_estimates: Dict[str, Dict[str, float]] = {}

    for config_idx, (name, k_bits, v_bits, metadata) in enumerate(quant_configs):
        print(f"Benchmarking quant config: {name}")
        result = run_one_config(
            config_name=name,
            prompts=benchmark_prompts,
            big_model=big_model,
            small_model=small_model,
            draft_steps=args.draft_steps,
            max_new_tokens=args.max_new_tokens,
            big_device=args.big_device,
            small_device=args.small_device,
            topk=args.topk,
            k_bits=k_bits,
            v_bits=v_bits,
            cuda_device_ids=cuda_device_ids,
            wandb_run=wandb_run,
            wandb_prefix="spec_kv",
            wandb_step_offset=config_idx * len(benchmark_prompts),
            shared_vocab_size=shared_vocab_size,
            key_quant_axis=args.key_quant_axis,
            key_group_size=args.key_group_size,
            key_residual_length=args.key_residual_length,
            target_token_references=benchmark_target_references,
            target_margin_references=benchmark_target_margins,
        )
        memory = estimate_total_kv_memory(
            big_model=big_model,
            small_model=small_model,
            big_dtype=args.big_dtype,
            small_dtype=args.small_dtype,
            seq_len=args.prompt_len + args.max_new_tokens,
            k_bits=k_bits,
            v_bits=v_bits,
            scale_bits=args.scale_bits,
            key_quant_axis=args.key_quant_axis,
            key_group_size=args.key_group_size,
            key_residual_length=args.key_residual_length,
        )
        summary = {**result["summary"], **memory}
        summary["metadata/spec"] = metadata.get("spec", name)
        summaries[name] = summary
        memory_estimates[name] = memory
        all_rows.extend(result["rows"])

        if wandb_run is not None:
            for key, value in memory.items():
                wandb_run.summary[f"spec_kv/{name}/{key}"] = value

    baseline = summaries.get("none")
    if baseline is not None:
        for name, summary in summaries.items():
            summary["accept_rate_delta_vs_none"] = summary.get("overall_accept_rate", 0.0) - baseline.get(
                "overall_accept_rate", 0.0
            )
            summary["top1_delta_vs_none"] = summary.get("round_top1_match", 0.0) - baseline.get("round_top1_match", 0.0)
            summary["js_delta_vs_none"] = summary.get("round_js", 0.0) - baseline.get("round_js", 0.0)
            if wandb_run is not None:
                wandb_run.summary[f"spec_kv/{name}/accept_rate_delta_vs_none"] = summary["accept_rate_delta_vs_none"]
                wandb_run.summary[f"spec_kv/{name}/top1_delta_vs_none"] = summary["top1_delta_vs_none"]
                wandb_run.summary[f"spec_kv/{name}/js_delta_vs_none"] = summary["js_delta_vs_none"]

    summary_payload = {
        "config": vars(args),
        "runtime": {
            "evaluator_version": "cached_dynamic_v4",
            "target_cache_reused": True,
            "draft_cache_reused": True,
            "cache_crop_mode": "in_place",
            "key_quant_axis": args.key_quant_axis,
            "key_group_size": args.key_group_size,
            "key_residual_length": args.key_residual_length,
            "quantization_update_mode": "prefill_once_then_new_tokens_only",
            "target_reference_generation_in_timing": False,
            "exactness_margin_mode": "minimum_of_tokenwise_reference_and_batched_verifier",
            "quantization_mode": "fake_quantized_values_with_estimated_packed_bytes",
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
        },
        "num_prompts": len(benchmark_prompts),
        "warmup_prompts": len(warmup_prompts),
        "quant_configs": [name for name, _, _, _ in quant_configs],
        "tokenizer_compatibility": compatibility,
        "shared_vocab_size": shared_vocab_size,
        "model_output_vocab_sizes": {
            "target": int(big_model.config.vocab_size),
            "draft": int(small_model.config.vocab_size),
        },
        "summaries": summaries,
        "memory_estimates": memory_estimates,
    }
    write_csv(all_rows, os.path.join(args.out_dir, "benchmark_rows.csv"))
    write_json(summary_payload, os.path.join(args.out_dir, "summary.json"))

    if wandb_run is not None:
        wandb_run.summary["num_prompts"] = len(benchmark_prompts)
        wandb_run.summary["warmup_prompts"] = len(warmup_prompts)

    print("Done!")
    print(f"  {os.path.join(args.out_dir, 'benchmark_rows.csv')}")
    print(f"  {os.path.join(args.out_dir, 'summary.json')}")
    for name, summary in summaries.items():
        print(
            f"  {name}: accept={summary['overall_accept_rate']:.4f} "
            f"draft_saved={100.0 * summary['draft_cache_saved_fraction']:.2f}% "
            f"total_saved={100.0 * summary['total_cache_saved_fraction']:.2f}% "
            f"js={summary.get('round_js', float('nan')):.5f}"
        )


if __name__ == "__main__":
    main()
