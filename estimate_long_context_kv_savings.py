#!/usr/bin/env python3
"""
Estimate long-context KV-cache memory for native SpecDec, KV Reduce prefix sharing,
draft KV quantization, and a simple MLA-style draft-cache baseline.

This script is intentionally analytical: it loads model configs only, so it can project the memory
story at 1k-32k context without needing to allocate the models.
"""

import argparse
import csv
import os
from typing import Any, Dict, List, Sequence

from transformers import AutoConfig

from eval_absorbed_spec_decode import parse_shared_layer_spec
from kv_utils import get_head_dim, get_num_kv_heads, write_json


def parse_csv_ints(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_items(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def dtype_num_bytes(dtype_name: str) -> int:
    normalized = dtype_name.lower()
    if normalized in {"bf16", "bfloat16", "fp16", "float16", "half"}:
        return 2
    if normalized in {"fp32", "float32"}:
        return 4
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def kv_cache_bytes(
    *,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    seq_len: int,
    bytes_per_elem: int,
    components_per_layer: float = 2.0,
) -> float:
    return float(num_layers * components_per_layer * num_kv_heads * head_dim * seq_len * bytes_per_elem)


def mib(num_bytes: float) -> float:
    return float(num_bytes / (1024.0**2))


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


def maybe_write_plot(rows: List[Dict[str, Any]], out_dir: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    specs = sorted({str(row["shared_layers_spec"]) for row in rows})
    plt.figure(figsize=(10, 6))
    for spec in specs:
        spec_rows = [row for row in rows if row["shared_layers_spec"] == spec]
        spec_rows.sort(key=lambda row: int(row["context_len"]))
        plt.plot(
            [row["context_len"] for row in spec_rows],
            [100.0 * row["kv_reduce_saved_fraction"] for row in spec_rows],
            marker="o",
            label=f"KV Reduce {spec}",
        )
    plt.xscale("log", base=2)
    plt.xlabel("Context length")
    plt.ylabel("Total KV-cache memory saved vs native SpecDec (%)")
    plt.title("KV Reduce Savings Grow With Long Prefixes")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "long_context_kv_savings.png"), dpi=200)
    plt.close()


def build_rows(args: argparse.Namespace) -> List[Dict[str, Any]]:
    target_config = AutoConfig.from_pretrained(args.big_model)
    draft_config = AutoConfig.from_pretrained(args.small_model)
    target_layers = int(target_config.num_hidden_layers)
    draft_layers = int(draft_config.num_hidden_layers)
    target_bytes_per_elem = dtype_num_bytes(args.big_dtype)
    draft_bytes_per_elem = dtype_num_bytes(args.small_dtype)
    shared_specs = parse_csv_items(args.shared_layers_specs)
    contexts = parse_csv_ints(args.contexts)
    quant_bits = parse_csv_ints(args.quant_bits)

    rows: List[Dict[str, Any]] = []
    for context_len in contexts:
        target_bytes = kv_cache_bytes(
            num_layers=target_layers,
            num_kv_heads=get_num_kv_heads(target_config),
            head_dim=get_head_dim(target_config),
            seq_len=context_len,
            bytes_per_elem=target_bytes_per_elem,
        )
        native_draft_bytes = kv_cache_bytes(
            num_layers=draft_layers,
            num_kv_heads=get_num_kv_heads(draft_config),
            head_dim=get_head_dim(draft_config),
            seq_len=context_len,
            bytes_per_elem=draft_bytes_per_elem,
        )
        native_total = target_bytes + native_draft_bytes

        quant_totals = {}
        for bits in quant_bits:
            quant_draft_bytes = native_draft_bytes * (bits / (8.0 * draft_bytes_per_elem))
            quant_totals[f"draft_kv_quant{bits}_total_mib"] = mib(target_bytes + quant_draft_bytes)
            quant_totals[f"draft_kv_quant{bits}_saved_fraction"] = (
                (native_total - (target_bytes + quant_draft_bytes)) / native_total if native_total > 0 else 0.0
            )

        mla_total = None
        mla_saved_fraction = None
        if args.mla_latent_dim > 0:
            rope_dim = max(0, int(args.mla_rope_dim))
            draft_mla_bytes = float(
                draft_layers
                * context_len
                * (int(args.mla_latent_dim) + rope_dim)
                * draft_bytes_per_elem
            )
            mla_total = target_bytes + draft_mla_bytes
            mla_saved_fraction = (native_total - mla_total) / native_total if native_total > 0 else 0.0

        for spec in shared_specs:
            shared_layers = len(parse_shared_layer_spec(spec, draft_layers))
            unshared_layers = draft_layers - shared_layers
            unshared_draft_bytes = kv_cache_bytes(
                num_layers=unshared_layers,
                num_kv_heads=get_num_kv_heads(draft_config),
                head_dim=get_head_dim(draft_config),
                seq_len=context_len,
                bytes_per_elem=draft_bytes_per_elem,
            )
            target_shared_prefix_read_bytes = kv_cache_bytes(
                num_layers=shared_layers,
                num_kv_heads=get_num_kv_heads(target_config),
                head_dim=get_head_dim(target_config),
                seq_len=context_len,
                bytes_per_elem=target_bytes_per_elem,
            )
            shared_tail_bytes = kv_cache_bytes(
                num_layers=shared_layers,
                num_kv_heads=get_num_kv_heads(draft_config),
                head_dim=get_head_dim(draft_config),
                seq_len=args.draft_tail_len,
                bytes_per_elem=draft_bytes_per_elem,
            )
            kv_reduce_draft_bytes = unshared_draft_bytes + shared_tail_bytes
            kv_reduce_total = target_bytes + kv_reduce_draft_bytes
            kv_reduce_hbm_read_proxy = unshared_draft_bytes + target_shared_prefix_read_bytes + shared_tail_bytes
            row = {
                "big_model": args.big_model,
                "small_model": args.small_model,
                "context_len": int(context_len),
                "draft_tail_len": int(args.draft_tail_len),
                "shared_layers_spec": spec,
                "num_shared_layers": int(shared_layers),
                "target_cache_mib": mib(target_bytes),
                "native_draft_cache_mib": mib(native_draft_bytes),
                "native_total_mib": mib(native_total),
                "kv_reduce_draft_cache_mib": mib(kv_reduce_draft_bytes),
                "kv_reduce_total_mib": mib(kv_reduce_total),
                "kv_reduce_saved_mib": mib(native_total - kv_reduce_total),
                "kv_reduce_saved_fraction": (native_total - kv_reduce_total) / native_total
                if native_total > 0
                else 0.0,
                "draft_cache_removed_fraction": (native_draft_bytes - kv_reduce_draft_bytes) / native_draft_bytes
                if native_draft_bytes > 0
                else 0.0,
                "native_draft_hbm_read_proxy_mib": mib(native_draft_bytes),
                "kv_reduce_hbm_read_proxy_mib": mib(kv_reduce_hbm_read_proxy),
                "hbm_read_proxy_delta_fraction": (kv_reduce_hbm_read_proxy - native_draft_bytes) / native_draft_bytes
                if native_draft_bytes > 0
                else 0.0,
                "draft_mla_total_mib": mib(mla_total) if mla_total is not None else None,
                "draft_mla_saved_fraction": mla_saved_fraction,
                **quant_totals,
            }
            rows.append(row)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analytical long-context KV-cache memory estimates.")
    parser.add_argument("--big_model", type=str, default="Qwen/Qwen2.5-3B")
    parser.add_argument("--small_model", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--big_dtype", type=str, default="bf16")
    parser.add_argument("--small_dtype", type=str, default="bf16")
    parser.add_argument("--contexts", type=str, default="512,1024,4096,8192,16384,32768")
    parser.add_argument("--shared_layers_specs", type=str, default="top:4,top:8,all")
    parser.add_argument("--draft_tail_len", type=int, default=4)
    parser.add_argument("--quant_bits", type=str, default="8,4")
    parser.add_argument("--mla_latent_dim", type=int, default=256)
    parser.add_argument(
        "--mla_rope_dim",
        type=int,
        default=128,
        help="Extra per-layer positional cache dimension for the draft MLA-style baseline.",
    )
    parser.add_argument("--out_dir", type=str, default="outputs/long_context_kv_savings")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rows = build_rows(args)
    write_csv(rows, os.path.join(args.out_dir, "kv_savings.csv"))
    write_json({"rows": rows, "config": vars(args)}, os.path.join(args.out_dir, "kv_savings.json"))
    maybe_write_plot(rows, args.out_dir)
    print("Done!")
    print(f"  {os.path.join(args.out_dir, 'kv_savings.csv')}")
    print(f"  {os.path.join(args.out_dir, 'kv_savings.json')}")
    plot_path = os.path.join(args.out_dir, "long_context_kv_savings.png")
    if os.path.exists(plot_path):
        print(f"  {plot_path}")


if __name__ == "__main__":
    main()
