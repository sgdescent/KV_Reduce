#!/usr/bin/env python3
"""
Evaluate target KV-cache quantization as the aggressive KV Reduce track.

This diagnostic does not implement a memory-saving kernel. Instead, it answers the
research question we need first:

  If the target prefix KV cache were stored at int8/int4/int3 and dequantized for
  attention, how much would target next-token behavior change?

For each sequence, we:
  1. Run the target model on a prefix and record native next-token logits.
  2. Recompute the same prefix through a cached path.
  3. Quantize/dequantize the prefix KV cache.
  4. Feed the last prefix token using the quantized cache and compare logits.

This simulates the numerical effect of compressed target KV while keeping the
implementation simple and independent of custom CUDA kernels.
"""

import argparse
import csv
import os
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

from kv_utils import (
    as_legacy_cache,
    distribution_metrics,
    iter_token_blocks,
    legacy_to_cache,
    load_causal_lm,
    load_tokenizer,
    set_seed,
    write_json,
)


def parse_csv_ints(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def dtype_bits(dtype_name: str) -> int:
    normalized = dtype_name.lower()
    if normalized in {"bf16", "bfloat16", "fp16", "float16", "half"}:
        return 16
    if normalized in {"fp32", "float32"}:
        return 32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def quantize_dequantize_per_vector_symmetric(x: torch.Tensor, bits: int) -> torch.Tensor:
    if bits >= 16:
        return x
    qmax = float((1 << (bits - 1)) - 1)
    # Per [batch, head, token] vector scaling keeps each head's direction stable.
    scale = x.float().abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / qmax
    q = torch.round(x.float() / scale).clamp(-qmax, qmax)
    return (q * scale).to(dtype=x.dtype)


def quantize_legacy_cache(
    legacy_cache: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
    bits: int,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
    return tuple(
        (
            quantize_dequantize_per_vector_symmetric(k, bits),
            quantize_dequantize_per_vector_symmetric(v, bits),
        )
        for k, v in legacy_cache
    )


def estimate_cache_bytes(
    legacy_cache: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
    *,
    value_bits: int,
    scale_bits: int,
) -> Tuple[float, float]:
    value_count = 0
    scale_count = 0
    for k, v in legacy_cache:
        value_count += k.numel() + v.numel()
        # One scale per [B, H, T] vector for K and V.
        scale_count += k.shape[0] * k.shape[1] * k.shape[2]
        scale_count += v.shape[0] * v.shape[1] * v.shape[2]
    full_bf16_bytes = float(value_count * 16 / 8)
    quantized_bytes = float(value_count * value_bits / 8 + scale_count * scale_bits / 8)
    return full_bf16_bytes, quantized_bytes


def mean_dict(rows: Iterable[Dict[str, float]]) -> Dict[str, float]:
    sums: Dict[str, float] = defaultdict(float)
    counts: Dict[str, int] = defaultdict(int)
    for row in rows:
        for key, value in row.items():
            sums[key] += float(value)
            counts[key] += 1
    return {key: sums[key] / max(1, counts[key]) for key in sums}


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate target KV-cache quantization drift.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--attn_implementation", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default="wikitext")
    parser.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--text_file", type=str, default=None)
    parser.add_argument("--text_column", type=str, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--split_fallbacks", type=str, default="validation,train")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--num_sequences", type=int, default=128)
    parser.add_argument("--prompt_len", type=int, default=1024)
    parser.add_argument("--bits", type=str, default="8,4,3")
    parser.add_argument("--scale_bits", type=int, default=16)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="outputs/target_cache_quant_eval")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="kv-reduce")
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    return parser


@torch.no_grad()
def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    bits_list = parse_csv_ints(args.bits)
    split_fallbacks = [item.strip() for item in args.split_fallbacks.split(",") if item.strip()]

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

    rows: List[Dict[str, Any]] = []
    per_bit_metrics: Dict[int, List[Dict[str, float]]] = {bits: [] for bits in bits_list}
    seq_iter = iter_token_blocks(
        tokenizer=tokenizer,
        seq_len=args.prompt_len + 1,
        max_blocks=args.num_sequences,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.split,
        text_file=args.text_file,
        text_column=args.text_column,
        shuffle=args.shuffle,
        seed=args.seed,
        streaming=args.streaming,
        split_fallbacks=split_fallbacks,
    )

    bar = tqdm(seq_iter, total=args.num_sequences, desc="Quant eval", unit="seq")
    for seq_idx, block in enumerate(bar):
        full = block.unsqueeze(0).to(args.device)
        context_ids = full[:, : args.prompt_len]
        label = full[:, args.prompt_len]
        cache_context_ids = context_ids[:, :-1]
        final_context_token = context_ids[:, -1:]

        native_out = model(input_ids=context_ids, use_cache=False)
        native_logits = native_out.logits[:, -1, :]
        native_nll = F.cross_entropy(native_logits.float(), label, reduction="none")

        prefix_out = model(input_ids=cache_context_ids, use_cache=True)
        legacy_cache = as_legacy_cache(prefix_out.past_key_values)
        full_cache_bytes, _ = estimate_cache_bytes(legacy_cache, value_bits=dtype_bits(args.dtype), scale_bits=0)

        for bits in bits_list:
            quant_legacy = quantize_legacy_cache(legacy_cache, bits)
            quant_cache = legacy_to_cache(quant_legacy)
            quant_out = model(input_ids=final_context_token, past_key_values=quant_cache, use_cache=False)
            quant_logits = quant_out.logits[:, -1, :]
            quant_nll = F.cross_entropy(quant_logits.float(), label, reduction="none")
            metrics = distribution_metrics(native_logits, quant_logits, topk=args.topk)
            _, quant_cache_bytes = estimate_cache_bytes(
                legacy_cache,
                value_bits=bits,
                scale_bits=args.scale_bits,
            )
            row = {
                "sequence_idx": seq_idx,
                "bits": bits,
                "native_nll": float(native_nll.mean().item()),
                "quantized_nll": float(quant_nll.mean().item()),
                "delta_nll": float((quant_nll - native_nll).mean().item()),
                "native_cache_mib": full_cache_bytes / (1024.0**2),
                "quantized_cache_mib": quant_cache_bytes / (1024.0**2),
                "cache_saved_fraction": (full_cache_bytes - quant_cache_bytes) / full_cache_bytes
                if full_cache_bytes
                else 0.0,
                **metrics,
            }
            rows.append(row)
            metric_row = {k: float(v) for k, v in row.items() if k not in {"sequence_idx", "bits"}}
            per_bit_metrics[bits].append(metric_row)
            if wandb_run is not None:
                wandb.log({f"quant/{bits}bit/{k}": v for k, v in metric_row.items()}, step=seq_idx)

    summary = {
        "config": vars(args),
        "bits": {},
    }
    for bits in bits_list:
        summary["bits"][str(bits)] = mean_dict(per_bit_metrics[bits])

    write_csv(rows, os.path.join(args.out_dir, "target_cache_quant_rows.csv"))
    write_json(summary, os.path.join(args.out_dir, "summary.json"))
    if wandb_run is not None:
        wandb_run.summary.update({f"{bits}bit/{k}": v for bits, metrics in summary["bits"].items() for k, v in metrics.items()})
        wandb_run.finish()

    print("Done!")
    for bits, metrics in summary["bits"].items():
        print(
            f"  {bits}bit: saved={100.0 * metrics['cache_saved_fraction']:.2f}% "
            f"top1={metrics['top1_match']:.4f} js={metrics['js']:.5f} "
            f"delta_nll={metrics['delta_nll']:.5f}"
        )
    print(f"  {os.path.join(args.out_dir, 'summary.json')}")
    print(f"  {os.path.join(args.out_dir, 'target_cache_quant_rows.csv')}")


if __name__ == "__main__":
    main()
