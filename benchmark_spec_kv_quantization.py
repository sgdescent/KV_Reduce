#!/usr/bin/env python3
"""
Benchmark sensitivity-aware draft KV-cache quantization inside speculative decoding.

The target verifier remains full precision, so final generation is still exact greedy
target decoding. Only the draft model's cached K/V tensors are fake-quantized between
cached decode steps. This lets us measure the useful research tradeoff:

  draft KV bytes saved vs. speculative acceptance retained.
"""

import argparse
import atexit
import csv
import os
import time
from typing import Any, Dict, List, Optional, Sequence

import torch

from kv_cache_quantization import (
    bit_allocation_stats,
    estimate_model_kv_cache_bytes,
    parse_quant_config_specs,
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


def quantize_cache_for_next_step(past_key_values, k_bits: Sequence[int], v_bits: Sequence[int]):
    legacy = as_legacy_cache(past_key_values)
    quantized = quantize_legacy_cache(legacy, k_bits, v_bits)
    return legacy_to_cache(quantized)


@torch.no_grad()
def target_next_logits(big_model, prefix_ids: torch.Tensor, big_device: str) -> torch.Tensor:
    out = big_model(input_ids=prefix_ids.to(big_device), use_cache=False)
    return out.logits[:, -1, :]


@torch.no_grad()
def greedy_target_generate(
    *,
    big_model,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    big_device: str,
    shared_vocab_size: int,
) -> List[int]:
    prefix = prompt_ids.clone()
    generated: List[int] = []
    for _ in range(max_new_tokens):
        logits = shared_token_logits(target_next_logits(big_model, prefix, big_device), shared_vocab_size)
        token = int(logits.argmax(dim=-1).item())
        generated.append(token)
        prefix = torch.cat([prefix, torch.tensor([[token]], dtype=prefix.dtype)], dim=1)
    return generated


@torch.no_grad()
def draft_first_logits_from_quantized_prefix(
    *,
    small_model,
    prefix_ids: torch.Tensor,
    small_device: str,
    k_bits: Sequence[int],
    v_bits: Sequence[int],
) -> Dict[str, Any]:
    prefix_on_device = prefix_ids.to(small_device)
    if prefix_on_device.shape[1] == 1:
        out = small_model(input_ids=prefix_on_device, use_cache=True)
        cache = quantize_cache_for_next_step(out.past_key_values, k_bits, v_bits)
        return {
            "logits": out.logits[:, -1, :],
            "cache": cache,
            "cache_len": int(prefix_on_device.shape[1]),
            "prefill_tokens": 1,
        }

    cache_context = prefix_on_device[:, :-1]
    current = prefix_on_device[:, -1:]
    context_out = small_model(input_ids=cache_context, use_cache=True)
    cache = quantize_cache_for_next_step(context_out.past_key_values, k_bits, v_bits)
    cache_len = int(cache_context.shape[1])
    out = small_model(
        input_ids=current,
        attention_mask=_ones_attention_mask(cache_len + 1, small_device),
        past_key_values=cache,
        use_cache=True,
        cache_position=_cache_position(cache_len, 1, small_device),
    )
    cache = quantize_cache_for_next_step(out.past_key_values, k_bits, v_bits)
    return {
        "logits": out.logits[:, -1, :],
        "cache": cache,
        "cache_len": cache_len + 1,
        "prefill_tokens": int(prefix_on_device.shape[1]),
    }


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
) -> Dict[str, Any]:
    input_ids = torch.tensor([[token]], dtype=dtype, device=small_device)
    out = small_model(
        input_ids=input_ids,
        attention_mask=_ones_attention_mask(cache_len + 1, small_device),
        past_key_values=cache,
        use_cache=True,
        cache_position=_cache_position(cache_len, 1, small_device),
    )
    cache = quantize_cache_for_next_step(out.past_key_values, k_bits, v_bits)
    return {"logits": out.logits[:, -1, :], "cache": cache, "cache_len": cache_len + 1}


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
) -> Dict[str, Any]:
    prefix = prompt_ids.clone()
    generated: List[int] = []
    proposed_tokens = 0
    accepted_tokens = 0
    target_calls = 0
    target_verify_calls = 0
    draft_decode_calls = 0
    draft_prefill_calls = 0
    draft_prefill_tokens = 0
    full_accept_rounds = 0
    num_rounds = 0
    round_metric_rows: List[Dict[str, float]] = []

    while len(generated) < max_new_tokens:
        num_rounds += 1
        current_prefix = prefix.clone()
        proposal: List[int] = []

        target_prefix_logits = shared_token_logits(
            target_next_logits(big_model, current_prefix, big_device),
            shared_vocab_size,
        )
        target_calls += 1

        first = draft_first_logits_from_quantized_prefix(
            small_model=small_model,
            prefix_ids=current_prefix,
            small_device=small_device,
            k_bits=k_bits,
            v_bits=v_bits,
        )
        draft_prefill_calls += 1
        draft_prefill_tokens += int(first["prefill_tokens"])
        draft_logits = shared_token_logits(first["logits"], shared_vocab_size)
        draft_cache = first["cache"]
        draft_cache_len = int(first["cache_len"])

        max_round_steps = min(draft_steps, max_new_tokens - len(generated))
        for proposal_idx in range(max_round_steps):
            draft_decode_calls += 1
            if proposal_idx == 0:
                metrics = distribution_metrics(
                    target_prefix_logits.to(draft_logits.device),
                    draft_logits,
                    topk=topk,
                )
                round_metric_rows.append(metrics)

            token = int(draft_logits.argmax(dim=-1).item())
            proposal.append(token)
            current_prefix = torch.cat([current_prefix, torch.tensor([[token]], dtype=current_prefix.dtype)], dim=1)

            if proposal_idx + 1 < max_round_steps:
                next_step = draft_next_logits_from_cache(
                    small_model=small_model,
                    token=token,
                    cache=draft_cache,
                    cache_len=draft_cache_len,
                    small_device=small_device,
                    dtype=current_prefix.dtype,
                    k_bits=k_bits,
                    v_bits=v_bits,
                )
                draft_logits = shared_token_logits(next_step["logits"], shared_vocab_size)
                draft_cache = next_step["cache"]
                draft_cache_len = int(next_step["cache_len"])

        proposed_tokens += len(proposal)
        verify_ids = current_prefix.to(big_device)
        verify_out = big_model(input_ids=verify_ids, use_cache=False)
        target_calls += 1
        target_verify_calls += 1
        verify_logits = shared_token_logits(verify_out.logits, shared_vocab_size)

        base_idx = int(prefix.shape[1]) - 1
        accepted_this_round = 0
        for idx, token in enumerate(proposal):
            target_token = int(verify_logits[:, base_idx + idx, :].argmax(dim=-1).item())
            if target_token != token:
                break
            accepted_this_round += 1

        accepted_tokens += accepted_this_round
        if accepted_this_round == len(proposal):
            full_accept_rounds += 1

        if accepted_this_round > 0:
            accepted_tensor = torch.tensor([proposal[:accepted_this_round]], dtype=prefix.dtype)
            prefix = torch.cat([prefix, accepted_tensor], dim=1)
            generated.extend(proposal[:accepted_this_round])

        if len(generated) >= max_new_tokens:
            break

        if accepted_this_round < len(proposal):
            correction_logits = verify_logits[:, base_idx + accepted_this_round, :]
        else:
            correction_logits = verify_logits[:, -1, :]
        correction_token = int(correction_logits.argmax(dim=-1).item())
        prefix = torch.cat([prefix, torch.tensor([[correction_token]], dtype=prefix.dtype)], dim=1)
        generated.append(correction_token)

    round_metrics = aggregate_rows(round_metric_rows)
    return {
        "generated_tokens": generated[:max_new_tokens],
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
        **{f"allocation/{key}": value for key, value in bit_allocation_stats(k_bits, v_bits).items()},
    }


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
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    reset_cuda_peak(cuda_device_ids)
    sync_cuda(cuda_device_ids)
    start = time.perf_counter()

    for prompt_idx, prompt_ids in enumerate(prompts):
        sync_cuda(cuda_device_ids)
        prompt_start = time.perf_counter()
        target_tokens = greedy_target_generate(
            big_model=big_model,
            prompt_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            big_device=big_device,
            shared_vocab_size=shared_vocab_size,
        )
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
        )
        sync_cuda(cuda_device_ids)
        elapsed_s = time.perf_counter() - prompt_start
        generated_tokens = len(result["generated_tokens"])
        row = {
            "config": config_name,
            "prompt_idx": int(prompt_idx),
            "latency_s": float(elapsed_s),
            "generated_tokens": int(generated_tokens),
            "tokens_per_second": float(generated_tokens / elapsed_s) if elapsed_s > 0 else 0.0,
            "ms_per_generated_token": float(1000.0 * elapsed_s / generated_tokens) if generated_tokens > 0 else 0.0,
            "matches_target_greedy": float(result["generated_tokens"] == target_tokens),
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

    big_model = load_causal_lm(args.big_model, device=args.big_device, dtype_name=args.big_dtype, attn_implementation="eager")
    small_model = load_causal_lm(args.small_model, device=args.small_device, dtype_name=args.small_dtype, attn_implementation="eager")
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
