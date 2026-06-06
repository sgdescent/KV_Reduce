#!/usr/bin/env python3
"""
Estimate the high-savings KV Reduce design space.

The existing long-context estimator focuses on the exact cached-prefix method:
target full KV + partial/tiny draft KV. This script adds the missing ceiling
analysis and the aggressive target-compression variants we would need for much
larger savings.

Important distinction:
  - exact_original_target=True means final verification can remain bit-for-bit
    the original target model, because the full target KV cache is still stored.
  - exact_original_target=False means we are also compressing the target cache.
    This can save far more memory, but the verified model is now a compressed
    target unless the compression is proven lossless.
"""

import argparse
import csv
import os
from typing import Any, Dict, Iterable, List

from transformers import AutoConfig

from kv_utils import get_head_dim, get_num_kv_heads, write_json


def parse_csv_ints(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def dtype_bits(dtype_name: str) -> int:
    normalized = dtype_name.lower()
    if normalized in {"bf16", "bfloat16", "fp16", "float16", "half"}:
        return 16
    if normalized in {"fp32", "float32"}:
        return 32
    if normalized in {"fp8", "float8", "e4m3", "e5m2"}:
        return 8
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def bytes_from_bits(num_values: float, bits: float) -> float:
    return float(num_values * bits / 8.0)


def mib(num_bytes: float) -> float:
    return float(num_bytes / (1024.0**2))


def kv_values_per_token(config) -> int:
    return int(config.num_hidden_layers) * 2 * get_num_kv_heads(config) * get_head_dim(config)


def kv_values_for_layers(config, num_layers: int, seq_len: int) -> float:
    return float(num_layers * 2 * get_num_kv_heads(config) * get_head_dim(config) * seq_len)


def latent_values_for_model(config, seq_len: int, latent_dim: int, rope_dim: int) -> float:
    # MLA-style cache stores one content latent plus decoupled positional dimensions per layer.
    return float(int(config.num_hidden_layers) * seq_len * (int(latent_dim) + int(rope_dim)))


def add_common_metrics(row: Dict[str, Any], native_total_bytes: float, method_total_bytes: float) -> None:
    row["method_total_mib"] = mib(method_total_bytes)
    row["saved_mib"] = mib(native_total_bytes - method_total_bytes)
    row["saved_fraction"] = (native_total_bytes - method_total_bytes) / native_total_bytes if native_total_bytes else 0.0
    row["saved_percent"] = 100.0 * row["saved_fraction"]
    row["compression_ratio_vs_native"] = native_total_bytes / method_total_bytes if method_total_bytes else None


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


def maybe_write_plots(rows: List[Dict[str, Any]], out_dir: str, highlight_context: int) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    context_rows = [row for row in rows if int(row["context_len"]) == int(highlight_context)]
    if not context_rows:
        context_rows = [row for row in rows if int(row["context_len"]) == max(int(r["context_len"]) for r in rows)]

    exact_rows = [row for row in context_rows if row["exact_original_target"]]
    aggressive_rows = [row for row in context_rows if not row["exact_original_target"]]

    def short_label(row: Dict[str, Any]) -> str:
        label = str(row["method"])
        label = label.replace("Exact ", "")
        label = label.replace("Aggressive ", "")
        label = label.replace("KV Reduce ", "KVR ")
        label = label.replace("target ", "T ")
        label = label.replace("draft ", "D ")
        return label

    plot_rows = exact_rows + aggressive_rows
    plt.figure(figsize=(13, 6))
    colors = ["#244061" if row["exact_original_target"] else "#cf4458" for row in plot_rows]
    plt.bar([short_label(row) for row in plot_rows], [row["saved_percent"] for row in plot_rows], color=colors)
    plt.axhline(0, color="#222222", linewidth=0.8)
    plt.ylabel("Total KV-cache memory saved vs native SpecDec (%)")
    plt.title(f"KV Reduce High-Savings Design Space at {highlight_context:,} Tokens")
    plt.xticks(rotation=35, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "high_savings_design_space.png"), dpi=220)
    plt.close()

    grouped = {}
    for row in rows:
        grouped.setdefault(str(row["method"]), []).append(row)

    plt.figure(figsize=(11, 6))
    for method, method_rows in grouped.items():
        method_rows.sort(key=lambda row: int(row["context_len"]))
        linewidth = 2.8 if method in {"Exact KV Reduce no draft prefix", "Aggressive target int4 + no draft prefix"} else 1.7
        plt.plot(
            [row["context_len"] for row in method_rows],
            [row["saved_percent"] for row in method_rows],
            marker="o",
            linewidth=linewidth,
            label=method,
        )
    plt.xscale("log", base=2)
    plt.xlabel("Context length")
    plt.ylabel("Total KV-cache memory saved vs native SpecDec (%)")
    plt.title("Tail KV Becomes Negligible at Long Context")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "high_savings_vs_context.png"), dpi=220)
    plt.close()


def build_rows(args: argparse.Namespace) -> List[Dict[str, Any]]:
    target_config = AutoConfig.from_pretrained(args.big_model)
    draft_config = AutoConfig.from_pretrained(args.small_model)
    contexts = parse_csv_ints(args.contexts)
    target_dtype_bits = dtype_bits(args.big_dtype)
    draft_dtype_bits = dtype_bits(args.small_dtype)
    quant_bits = parse_csv_ints(args.target_quant_bits)
    mla_latent_dims = parse_csv_ints(args.target_mla_latent_dims)

    target_layers = int(target_config.num_hidden_layers)
    draft_layers = int(draft_config.num_hidden_layers)
    target_values_per_token = kv_values_per_token(target_config)
    draft_values_per_token = kv_values_per_token(draft_config)
    draft_tail_layers = draft_layers if args.draft_tail_layers < 0 else min(draft_layers, int(args.draft_tail_layers))

    rows: List[Dict[str, Any]] = []
    for context_len in contexts:
        target_full_bytes = bytes_from_bits(target_values_per_token * context_len, target_dtype_bits)
        draft_full_bytes = bytes_from_bits(draft_values_per_token * context_len, draft_dtype_bits)
        native_total_bytes = target_full_bytes + draft_full_bytes
        draft_tail_bytes = bytes_from_bits(
            kv_values_for_layers(draft_config, draft_tail_layers, int(args.draft_tail_len)),
            draft_dtype_bits,
        )
        common = {
            "big_model": args.big_model,
            "small_model": args.small_model,
            "context_len": int(context_len),
            "draft_tail_len": int(args.draft_tail_len),
            "target_layers": target_layers,
            "draft_layers": draft_layers,
            "target_kv_values_per_token": target_values_per_token,
            "draft_kv_values_per_token": draft_values_per_token,
            "native_target_mib": mib(target_full_bytes),
            "native_draft_mib": mib(draft_full_bytes),
            "native_total_mib": mib(native_total_bytes),
            "native_draft_fraction_of_total": draft_full_bytes / native_total_bytes if native_total_bytes else 0.0,
        }

        exact_no_draft = dict(common)
        exact_no_draft.update(
            {
                "method": "Exact KV Reduce no draft prefix",
                "family": "exact",
                "exact_original_target": True,
                "description": "Store the original target KV cache and only a tiny speculative draft tail cache.",
            }
        )
        exact_total = target_full_bytes + draft_tail_bytes
        add_common_metrics(exact_no_draft, native_total_bytes, exact_total)
        exact_no_draft["draft_prefix_removed_fraction"] = (draft_full_bytes - draft_tail_bytes) / draft_full_bytes
        rows.append(exact_no_draft)

        draft_int4 = dict(common)
        draft_int4.update(
            {
                "method": "Exact native target + draft int4",
                "family": "exact",
                "exact_original_target": True,
                "description": "Keep original target cache; quantize the draft cache but do not share it.",
            }
        )
        draft_int4_total = target_full_bytes + bytes_from_bits(draft_values_per_token * context_len, 4)
        add_common_metrics(draft_int4, native_total_bytes, draft_int4_total)
        rows.append(draft_int4)

        exact_no_draft_plus_tail_int4 = dict(common)
        exact_no_draft_plus_tail_int4.update(
            {
                "method": "Exact KV Reduce + int4 tail",
                "family": "exact",
                "exact_original_target": True,
                "description": "Store original target cache; remove draft prefix; quantize the tiny draft tail.",
            }
        )
        int4_tail_total = target_full_bytes + bytes_from_bits(
            kv_values_for_layers(draft_config, draft_tail_layers, int(args.draft_tail_len)),
            4,
        )
        add_common_metrics(exact_no_draft_plus_tail_int4, native_total_bytes, int4_tail_total)
        rows.append(exact_no_draft_plus_tail_int4)

        for bits in quant_bits:
            quant = dict(common)
            quant.update(
                {
                    "method": f"Aggressive target int{bits} + no draft prefix",
                    "family": "target_quantized",
                    "target_cache_bits": int(bits),
                    "exact_original_target": False,
                    "description": "Quantize target KV itself and remove the draft prefix. High savings; target is compressed.",
                }
            )
            target_quant_total = bytes_from_bits(target_values_per_token * context_len, bits) + draft_tail_bytes
            add_common_metrics(quant, native_total_bytes, target_quant_total)
            rows.append(quant)

        for latent_dim in mla_latent_dims:
            mla = dict(common)
            mla.update(
                {
                    "method": f"Aggressive target MLA c{latent_dim}+r{args.target_mla_rope_dim}",
                    "family": "target_mla",
                    "target_mla_latent_dim": int(latent_dim),
                    "target_mla_rope_dim": int(args.target_mla_rope_dim),
                    "exact_original_target": False,
                    "description": "Convert the target to an MLA-style latent KV cache and remove draft prefix.",
                }
            )
            target_mla_values = latent_values_for_model(
                target_config,
                context_len,
                int(latent_dim),
                int(args.target_mla_rope_dim),
            )
            target_mla_total = bytes_from_bits(target_mla_values, target_dtype_bits) + draft_tail_bytes
            add_common_metrics(mla, native_total_bytes, target_mla_total)
            rows.append(mla)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Estimate high-savings KV Reduce variants.")
    parser.add_argument("--big_model", type=str, default="Qwen/Qwen2.5-7B")
    parser.add_argument("--small_model", type=str, default="Qwen/Qwen2.5-3B")
    parser.add_argument("--big_dtype", type=str, default="bf16")
    parser.add_argument("--small_dtype", type=str, default="bf16")
    parser.add_argument("--contexts", type=str, default="1024,4096,8192,16384,32768,65536")
    parser.add_argument("--draft_tail_len", type=int, default=4)
    parser.add_argument(
        "--draft_tail_layers",
        type=int,
        default=-1,
        help="Number of draft layers that keep a tail cache. -1 means all draft layers.",
    )
    parser.add_argument("--target_quant_bits", type=str, default="8,4,3")
    parser.add_argument("--target_mla_latent_dims", type=str, default="128,256,512")
    parser.add_argument("--target_mla_rope_dim", type=int, default=64)
    parser.add_argument("--plot_context", type=int, default=32768)
    parser.add_argument("--out_dir", type=str, default="outputs/high_savings_design_space")
    return parser


def print_summary(rows: Iterable[Dict[str, Any]], plot_context: int) -> None:
    context_rows = [row for row in rows if int(row["context_len"]) == int(plot_context)]
    if not context_rows:
        context_rows = list(rows)
    context_rows = sorted(context_rows, key=lambda row: row["saved_fraction"], reverse=True)
    print(f"Top design points at context={plot_context}:")
    for row in context_rows[:8]:
        exact = "exact-original" if row["exact_original_target"] else "compressed-target"
        print(
            f"  {row['saved_percent']:6.2f}% saved | "
            f"{row['compression_ratio_vs_native']:.2f}x smaller | {exact:17s} | {row['method']}"
        )


def main() -> None:
    args = build_parser().parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rows = build_rows(args)
    write_csv(rows, os.path.join(args.out_dir, "high_savings_design_space.csv"))
    write_json({"config": vars(args), "rows": rows}, os.path.join(args.out_dir, "high_savings_design_space.json"))
    maybe_write_plots(rows, args.out_dir, args.plot_context)
    print_summary(rows, args.plot_context)
    print("Done!")
    print(f"  {os.path.join(args.out_dir, 'high_savings_design_space.csv')}")
    print(f"  {os.path.join(args.out_dir, 'high_savings_design_space.json')}")
    for plot_name in ["high_savings_design_space.png", "high_savings_vs_context.png"]:
        plot_path = os.path.join(args.out_dir, plot_name)
        if os.path.exists(plot_path):
            print(f"  {plot_path}")


if __name__ == "__main__":
    main()
