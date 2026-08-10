#!/usr/bin/env python3
"""Measure free-running greedy drift from standalone KV-cache quantization."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from tqdm import tqdm

EVALUATOR_VERSION = "free_running_cached_v1"
PER_TOKEN_AXIS = "per_token"
PER_CHANNEL_AXIS = "per_channel"
SYMMETRIC_QUANT = "symmetric"
AFFINE_QUANT = "affine"


def parse_csv_items(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def write_csv(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean_numeric_rows(
    rows: Iterable[Dict[str, Any]],
    *,
    exclude: Sequence[str] = (),
) -> Dict[str, float]:
    excluded = set(exclude)
    values: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        for key, value in row.items():
            if key in excluded or isinstance(value, (str, bool)) or value is None:
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if numeric == numeric:
                values[key].append(numeric)
    return {key: sum(items) / len(items) for key, items in values.items() if items}


def rollout_comparison(
    reference_tokens: Sequence[int],
    candidate_tokens: Sequence[int],
    reference_margins: Sequence[float],
    *,
    tie_margin: float,
) -> Dict[str, Any]:
    compared = min(len(reference_tokens), len(candidate_tokens))
    matched = sum(
        int(reference_tokens[idx] == candidate_tokens[idx]) for idx in range(compared)
    )
    first = next(
        (
            idx
            for idx, (reference, candidate) in enumerate(
                zip(reference_tokens, candidate_tokens)
            )
            if reference != candidate
        ),
        min(len(reference_tokens), len(candidate_tokens)),
    )
    exact = list(reference_tokens) == list(candidate_tokens)
    first_margin: Optional[float] = None
    if not exact and first < len(reference_margins):
        first_margin = float(reference_margins[first])
    denominator = max(1, max(len(reference_tokens), len(candidate_tokens)))
    return {
        "exact_sequence_match": float(exact),
        "token_match_fraction": float(matched / denominator),
        "prefix_match_tokens": float(first),
        "prefix_retained_fraction": float(first / denominator),
        "first_divergence_position": -1 if exact else int(first),
        "reference_margin_at_first_divergence": first_margin,
        "first_divergence_is_bf16_tie": float(
            first_margin is not None and first_margin <= tie_margin
        ),
    }


@torch.inference_mode()
def rollout_from_prefill(
    *,
    model,
    prefix_logits: torch.Tensor,
    prefix_legacy_cache: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
    prefix_len: int,
    max_new_tokens: int,
    device: str,
    k_bits: Sequence[int],
    v_bits: Sequence[int],
    key_quant_axis: str,
    key_group_size: int,
    key_residual_length: int,
    value_quant_scheme: str,
) -> Dict[str, Any]:
    from benchmark_spec_kv_quantization import (
        cached_step,
        quantize_cache_for_next_step,
        top1_logit_margin,
    )
    from kv_utils import clone_legacy_cache, legacy_to_cache

    cache = legacy_to_cache(clone_legacy_cache(prefix_legacy_cache))
    cache = quantize_cache_for_next_step(
        cache,
        k_bits,
        v_bits,
        key_quant_axis=key_quant_axis,
        key_group_size=key_group_size,
        key_residual_length=key_residual_length,
        value_quant_scheme=value_quant_scheme,
    )
    logits = prefix_logits
    cache_len = int(prefix_len)
    tokens: List[int] = []
    margins: List[float] = []

    if device.startswith("cuda"):
        torch.cuda.synchronize(torch.device(device))
    start = time.perf_counter()
    for token_idx in range(max_new_tokens):
        margins.append(top1_logit_margin(logits))
        token = int(logits.argmax(dim=-1).item())
        tokens.append(token)
        if token_idx + 1 >= max_new_tokens:
            break
        step = cached_step(
            model=model,
            input_ids=torch.tensor([[token]], dtype=torch.long),
            cache=cache,
            cache_len=cache_len,
            device=device,
        )
        logits = step["logits"][:, -1, :]
        cache = step["cache"]
        cache_len = int(step["cache_len"])
        cache = quantize_cache_for_next_step(
            cache,
            k_bits,
            v_bits,
            new_tokens=1,
            key_quant_axis=key_quant_axis,
            key_group_size=key_group_size,
            key_residual_length=key_residual_length,
            value_quant_scheme=value_quant_scheme,
        )
    if device.startswith("cuda"):
        torch.cuda.synchronize(torch.device(device))
    elapsed = time.perf_counter() - start
    return {
        "tokens": tokens,
        "top1_margins": margins,
        "elapsed_s": float(elapsed),
        "tokens_per_second_fake_quant": float(len(tokens) / elapsed) if elapsed > 0 else 0.0,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--attn_implementation", default=None)
    parser.add_argument("--dataset_name", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--dataset_config", default="sample-10BT")
    parser.add_argument("--text_file", default=None)
    parser.add_argument("--text_column", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--split_fallbacks", default="validation,test")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--skip_prompts", type=int, default=0)
    parser.add_argument("--num_prompts", type=int, default=32)
    parser.add_argument("--prompt_len", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument(
        "--quant_configs",
        default="none;k8v4;k4v8;k4v4;k3v4;k4v3;k2v4;k4v2",
    )
    parser.add_argument("--scale_bits", type=int, default=16)
    parser.add_argument(
        "--key_quant_axis",
        choices=[PER_TOKEN_AXIS, PER_CHANNEL_AXIS],
        default=PER_CHANNEL_AXIS,
    )
    parser.add_argument("--key_group_size", type=int, default=32)
    parser.add_argument("--key_residual_length", type=int, default=128)
    parser.add_argument(
        "--value_quant_scheme",
        choices=[SYMMETRIC_QUANT, AFFINE_QUANT],
        default=AFFINE_QUANT,
    )
    parser.add_argument("--bf16_tie_margin", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", default="outputs/kivi_free_generation")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default="kv-reduce")
    parser.add_argument("--wandb_group", default="kivi-free-generation")
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--wandb_entity", default=None)
    return parser


@torch.inference_mode()
def main() -> None:
    from benchmark_spec_kv_quantization import cached_prefill
    from kv_cache_quantization import estimate_model_kv_cache_bytes, parse_quant_config_specs
    from kv_utils import (
        as_legacy_cache,
        clone_legacy_cache,
        iter_token_blocks,
        load_causal_lm,
        load_tokenizer,
        set_seed,
        write_json,
    )

    args = build_parser().parse_args()
    if args.num_prompts <= 0 or args.prompt_len <= 0 or args.max_new_tokens <= 0:
        raise ValueError("num_prompts, prompt_len, and max_new_tokens must be positive.")
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wandb_run = None
    if args.wandb:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            name=args.wandb_run_name,
            config=vars(args),
        )

    tokenizer = load_tokenizer(args.model)
    model = load_causal_lm(
        args.model,
        device=args.device,
        dtype_name=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    num_layers = int(model.config.num_hidden_layers)
    configs = parse_quant_config_specs(args.quant_configs, num_layers)
    prompts = list(
        iter_token_blocks(
            tokenizer=tokenizer,
            seq_len=args.prompt_len,
            max_blocks=args.num_prompts,
            dataset_name=args.dataset_name,
            dataset_config=args.dataset_config,
            split=args.split,
            text_file=args.text_file,
            text_column=args.text_column,
            shuffle=args.shuffle,
            seed=args.seed,
            streaming=args.streaming,
            split_fallbacks=parse_csv_items(args.split_fallbacks),
            skip_blocks=args.skip_prompts,
        )
    )
    if len(prompts) != args.num_prompts:
        raise RuntimeError(f"Requested {args.num_prompts} prompts but loaded {len(prompts)}.")

    rows: List[Dict[str, Any]] = []
    progress = tqdm(prompts, desc="Free-running prompts", unit="prompt")
    for prompt_idx, prompt in enumerate(progress):
        prefill = cached_prefill(model, prompt.unsqueeze(0), args.device)
        prefix_logits = prefill["logits"]
        prefix_legacy = clone_legacy_cache(as_legacy_cache(prefill["cache"]))

        reference_name, reference_k_bits, reference_v_bits, _ = next(
            (config for config in configs if config[0] == "none"),
            ("none", [16] * num_layers, [16] * num_layers, {}),
        )
        reference = rollout_from_prefill(
            model=model,
            prefix_logits=prefix_logits,
            prefix_legacy_cache=prefix_legacy,
            prefix_len=args.prompt_len,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
            k_bits=reference_k_bits,
            v_bits=reference_v_bits,
            key_quant_axis=args.key_quant_axis,
            key_group_size=args.key_group_size,
            key_residual_length=args.key_residual_length,
            value_quant_scheme=args.value_quant_scheme,
        )

        for config_idx, (name, k_bits, v_bits, metadata) in enumerate(configs):
            candidate = reference if name == reference_name else rollout_from_prefill(
                model=model,
                prefix_logits=prefix_logits,
                prefix_legacy_cache=prefix_legacy,
                prefix_len=args.prompt_len,
                max_new_tokens=args.max_new_tokens,
                device=args.device,
                k_bits=k_bits,
                v_bits=v_bits,
                key_quant_axis=args.key_quant_axis,
                key_group_size=args.key_group_size,
                key_residual_length=args.key_residual_length,
                value_quant_scheme=args.value_quant_scheme,
            )
            memory = estimate_model_kv_cache_bytes(
                config=model.config,
                seq_len=args.prompt_len + args.max_new_tokens - 1,
                dtype_name=args.dtype,
                k_bits_by_layer=k_bits,
                v_bits_by_layer=v_bits,
                scale_bits=args.scale_bits,
                key_quant_axis=args.key_quant_axis,
                key_group_size=args.key_group_size,
                key_residual_length=args.key_residual_length,
                value_quant_scheme=args.value_quant_scheme,
            )
            row = {
                "prompt_idx": prompt_idx,
                "config": name,
                "model": args.model,
                "seed": args.seed,
                "skip_prompts": args.skip_prompts,
                "prompt_len": args.prompt_len,
                "max_new_tokens": args.max_new_tokens,
                **rollout_comparison(
                    reference["tokens"],
                    candidate["tokens"],
                    reference["top1_margins"],
                    tie_margin=args.bf16_tie_margin,
                ),
                "elapsed_s_fake_quant": candidate["elapsed_s"],
                "tokens_per_second_fake_quant": candidate["tokens_per_second_fake_quant"],
                "native_cache_bytes": memory["native_cache_bytes"],
                "quantized_cache_bytes": memory["quantized_cache_bytes"],
                "cache_saved_fraction": memory["cache_saved_fraction"],
                "key_quantized_prefix_tokens": memory["key_quantized_prefix_tokens"],
                "key_residual_tokens": memory["key_residual_tokens"],
                "reference_token_ids": json.dumps(reference["tokens"]),
                "candidate_token_ids": json.dumps(candidate["tokens"]),
                "allocation_source": metadata.get("allocation_path", "uniform"),
            }
            rows.append(row)
            if wandb_run is not None:
                numeric = {
                    f"generation/{name}/{key}": value
                    for key, value in row.items()
                    if isinstance(value, (int, float)) and not isinstance(value, bool)
                }
                wandb_run.log(numeric, step=prompt_idx * len(configs) + config_idx)
        del prefix_legacy, prefill
        torch.cuda.empty_cache()

    summaries = {
        name: mean_numeric_rows(
            [row for row in rows if row["config"] == name],
            exclude=("prompt_idx", "seed", "skip_prompts", "prompt_len", "max_new_tokens"),
        )
        for name, _, _, _ in configs
    }
    payload = {
        "config": vars(args),
        "runtime": {
            "evaluator_version": EVALUATOR_VERSION,
            "decode_mode": "cached_greedy_fake_quant",
            "production_throughput_claim": False,
            "reference": "bf16_cached_greedy",
        },
        "observed_prompts": len(prompts),
        "summaries": summaries,
    }
    write_csv(rows, out_dir / "rows.csv")
    write_json(payload, str(out_dir / "summary.json"))
    if wandb_run is not None:
        for name, summary in summaries.items():
            for key, value in summary.items():
                wandb_run.summary[f"generation_summary/{name}/{key}"] = value
        wandb_run.summary["runtime/evaluator_version"] = EVALUATOR_VERSION
        wandb_run.summary["runtime/production_throughput_claim"] = False
        wandb_run.finish()

    print("Done!")
    for name, summary in summaries.items():
        print(
            f"  {name}: exact={summary.get('exact_sequence_match', 0.0):.4f} "
            f"token_match={summary.get('token_match_fraction', 0.0):.4f} "
            f"prefix={summary.get('prefix_retained_fraction', 0.0):.4f} "
            f"saved={100.0 * summary.get('cache_saved_fraction', 0.0):.2f}%"
        )


if __name__ == "__main__":
    main()
