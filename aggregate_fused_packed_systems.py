#!/usr/bin/env python3
"""Aggregate a cross-architecture fused packed-KV attention benchmark matrix."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


EXPECTED_VERSION = "packed_fused_attention_v1"
EXPECTED_SHAPES = {
    "qwen25_15b",
    "qwen25_3b",
    "qwen3_4b",
    "llama32_3b",
    "olmo2_1b",
    "smollm2_360m",
}
EXPECTED_BATCHES = {1, 4, 16}


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def geometric_mean(values: Iterable[float]) -> float:
    positive = [float(value) for value in values if float(value) > 0.0]
    if not positive:
        raise ValueError("Geometric mean requires positive values.")
    return math.exp(sum(math.log(value) for value in positive) / len(positive))


def select_best_rows(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, int, int, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["shape"]),
            int(row["batch_size"]),
            int(row["context"]),
            str(row["config"]),
        )
        grouped[key].append(row)
    return [
        min(items, key=lambda item: float(item["fused_decode_median_ms"]))
        for _, items in sorted(grouped.items())
    ]


def collect(root: Path, *, require_complete: bool) -> Tuple[List[Dict[str, Any]], List[str]]:
    rows: List[Dict[str, Any]] = []
    observed: set[Tuple[str, int]] = set()
    sources: List[str] = []
    for path in sorted(root.glob("*/batch_*/summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        runtime = payload.get("runtime", {})
        if runtime.get("evaluator_version") != EXPECTED_VERSION:
            raise ValueError(f"Unexpected evaluator in {path}: {runtime}")
        if runtime.get("storage_mode") != "actual_bit_packed_uint8_payloads":
            raise ValueError(f"{path} does not use actual packed storage.")
        shape = path.parents[1].name
        config = payload["config"]
        batch_size = int(config["batch_size"])
        observed.add((shape, batch_size))
        sources.append(str(path))
        for raw in payload.get("rows", []):
            rows.append(
                {
                    "shape": shape,
                    "batch_size": batch_size,
                    "query_heads": int(config["query_heads"]),
                    "kv_heads": int(config["kv_heads"]),
                    "head_dim": int(config["head_dim"]),
                    "num_layers": int(config["num_layers"]),
                    **raw,
                }
            )
    expected = {(shape, batch) for shape in EXPECTED_SHAPES for batch in EXPECTED_BATCHES}
    missing = sorted(expected - observed)
    if require_complete and missing:
        raise ValueError(f"Packed systems matrix is incomplete: {missing}")
    if not rows:
        raise ValueError(f"No packed systems summaries found under {root}.")
    return rows, sources


def make_plot(rows: Sequence[Dict[str, Any]], out_dir: Path) -> List[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    subset = [row for row in rows if row["config"] == "k4v4"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=True)
    for axis, batch_size in zip(axes, sorted(EXPECTED_BATCHES)):
        batch_rows = [row for row in subset if int(row["batch_size"]) == batch_size]
        for shape in sorted({str(row["shape"]) for row in batch_rows}):
            shape_rows = sorted(
                [row for row in batch_rows if row["shape"] == shape],
                key=lambda row: int(row["context"]),
            )
            axis.plot(
                [int(row["context"]) for row in shape_rows],
                [float(row["fused_speedup_vs_native"]) for row in shape_rows],
                marker="o",
                label=shape,
            )
        axis.axhline(1.0, color="black", linewidth=1, linestyle="--")
        axis.set_xscale("log", base=2)
        axis.set_title(f"Batch {batch_size}")
        axis.set_xlabel("Context length")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Best packed speedup vs BF16 SDPA")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8)
    fig.suptitle("Fused K4V4 Decode Across Serving Shapes", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    paths = []
    for extension in ("png", "pdf"):
        path = out_dir / f"fused_packed_systems.{extension}"
        fig.savefig(path, dpi=240, bbox_inches="tight")
        paths.append(str(path))
    plt.close(fig)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--require_complete", action="store_true")
    args = parser.parse_args()
    rows, sources = collect(args.root, require_complete=args.require_complete)
    best = select_best_rows(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "all_kernel_rows.csv", rows)
    write_csv(args.out_dir / "best_kernel_rows.csv", best)
    summaries = []
    for config in sorted({str(row["config"]) for row in best}):
        config_rows = [row for row in best if row["config"] == config]
        summaries.append(
            {
                "config": config,
                "num_cells": len(config_rows),
                "geomean_speedup_vs_native": geometric_mean(
                    row["fused_speedup_vs_native"] for row in config_rows
                ),
                "geomean_speedup_vs_unfused": geometric_mean(
                    row["fused_speedup_vs_unfused"] for row in config_rows
                ),
                "fraction_faster_than_native": sum(
                    float(row["fused_speedup_vs_native"]) > 1.0 for row in config_rows
                )
                / len(config_rows),
                "mean_cache_saved_fraction": sum(
                    float(row["cache_saved_fraction"]) for row in config_rows
                )
                / len(config_rows),
                "worst_kernel_output_max_abs_error": max(
                    float(row["kernel_output_max_abs_error"]) for row in config_rows
                ),
            }
        )
    write_csv(args.out_dir / "config_summary.csv", summaries)
    payload = {
        "runtime": {
            "source_evaluator_version": EXPECTED_VERSION,
            "storage_mode": "actual_bit_packed_uint8_payloads",
            "kernel_scope": "single_token_gqa_decode_microbenchmark",
            "production_throughput_claim": False,
            "complete_matrix_gate": args.require_complete,
        },
        "num_source_runs": len(sources),
        "num_kernel_rows": len(rows),
        "num_best_cells": len(best),
        "sources": sources,
        "summaries": summaries,
        "plots": make_plot(best, args.out_dir),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(args.out_dir / "summary.json")


if __name__ == "__main__":
    main()
