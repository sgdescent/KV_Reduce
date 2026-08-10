#!/usr/bin/env python3
"""Compare equal-memory K/V asymmetry across naive and KIVI quantizers."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

from spec_kv_statistics import bootstrap_mean_ci


PAIR_LABELS = {
    "qwen25_3b_15b": "Qwen2.5 3B/1.5B",
    "qwen25_7b_3b": "Qwen2.5 7B/3B",
    "qwen3_8b_4b": "Qwen3 8B/4B",
    "llama31_8b_llama32_3b": "Llama 3.1/3.2",
    "olmo2_7b_1b": "OLMo 2 7B/1B",
    "smollm2_17b_360m": "SmolLM2 1.7B/360M",
}


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


def load_naive_rows(path: Path, *, run: str) -> Dict[str, Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    output = {}
    for row in payload.get("comparisons", []):
        if row.get("run") != run:
            continue
        pair = str(row["pair"])
        output[pair] = {
            "effect": float(row["accept_rate_delta"]),
            "ci_low": float(row["delta_ci_low"]),
            "ci_high": float(row["delta_ci_high"]),
            "memory_saved_fraction": float(row["total_cache_saved_fraction"]),
        }
    return output


def load_kivi_rows(root: Path, *, context: int) -> Dict[str, Dict[str, Any]]:
    output = {}
    for summary_path in sorted(root.glob("*/summary.json")):
        pair = summary_path.parent.name
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        for row in payload.get("comparisons", []):
            if int(row["context"]) != context:
                continue
            if row["config_a"] != "k8v4" or row["config_b"] != "k4v8":
                continue
            output[pair] = {
                "effect": float(row["spec_acceptance_a_minus_b_mean"]),
                "ci_low": float(row["spec_acceptance_a_minus_b_ci_low"]),
                "ci_high": float(row["spec_acceptance_a_minus_b_ci_high"]),
                "memory_saved_fraction": 0.5
                * (
                    float(row["config_a_total_saved_fraction"])
                    + float(row["config_b_total_saved_fraction"])
                ),
                "memory_gap": float(row["absolute_total_saved_fraction_gap"]),
            }
    return output


def join_geometry_rows(
    naive: Dict[str, Dict[str, Any]],
    kivi: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    output = []
    for pair in sorted(set(naive) & set(kivi)):
        naive_row = naive[pair]
        kivi_row = kivi[pair]
        output.append(
            {
                "pair": pair,
                "pair_label": PAIR_LABELS.get(pair, pair),
                "naive_k8v4_minus_k4v8": naive_row["effect"],
                "naive_ci_low": naive_row["ci_low"],
                "naive_ci_high": naive_row["ci_high"],
                "kivi_k8v4_minus_k4v8": kivi_row["effect"],
                "kivi_ci_low": kivi_row["ci_low"],
                "kivi_ci_high": kivi_row["ci_high"],
                "geometry_shift_kivi_minus_naive": kivi_row["effect"] - naive_row["effect"],
                "naive_total_saved_fraction": naive_row["memory_saved_fraction"],
                "kivi_total_saved_fraction": kivi_row["memory_saved_fraction"],
                "kivi_equal_memory_gap": kivi_row.get("memory_gap", 0.0),
            }
        )
    return output


def make_plot(rows: Sequence[Dict[str, Any]], out_dir: Path) -> List[str]:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return []
    labels = [str(row["pair_label"]) for row in rows]
    x = np.arange(len(rows))
    width = 0.36
    naive = [100.0 * float(row["naive_k8v4_minus_k4v8"]) for row in rows]
    kivi = [100.0 * float(row["kivi_k8v4_minus_k4v8"]) for row in rows]
    fig, axis = plt.subplots(figsize=(10.2, 4.8))
    axis.bar(x - width / 2, naive, width, label="Per-token symmetric", color="#C84C4C")
    axis.bar(x + width / 2, kivi, width, label="KIVI geometry", color="#187F78")
    axis.axhline(0.0, color="#222222", linewidth=1)
    axis.set_xticks(x, labels, rotation=25, ha="right")
    axis.set_ylabel("Acceptance: K8V4 minus K4V8 (pp)")
    axis.set_title("K/V Precision Asymmetry Depends on Quantizer Geometry", fontweight="bold")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    paths = []
    for extension in ("png", "pdf"):
        path = out_dir / f"quantizer_geometry_asymmetry.{extension}"
        fig.savefig(path, dpi=240, bbox_inches="tight")
        paths.append(str(path))
    plt.close(fig)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--naive_summary", type=Path, required=True)
    parser.add_argument("--kivi_comparison_root", type=Path, required=True)
    parser.add_argument("--naive_run", default="wikitext_ctx1024")
    parser.add_argument("--context", type=int, default=1024)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--bootstrap_samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = join_geometry_rows(
        load_naive_rows(args.naive_summary, run=args.naive_run),
        load_kivi_rows(args.kivi_comparison_root, context=args.context),
    )
    if not rows:
        raise ValueError("No model pairs were shared by the two quantizer campaigns.")
    naive_macro = bootstrap_mean_ci(
        [float(row["naive_k8v4_minus_k4v8"]) for row in rows],
        seed=args.seed,
        samples=args.bootstrap_samples,
    )
    kivi_macro = bootstrap_mean_ci(
        [float(row["kivi_k8v4_minus_k4v8"]) for row in rows],
        seed=args.seed + 1,
        samples=args.bootstrap_samples,
    )
    shift_macro = bootstrap_mean_ci(
        [float(row["geometry_shift_kivi_minus_naive"]) for row in rows],
        seed=args.seed + 2,
        samples=args.bootstrap_samples,
    )
    write_csv(args.out_dir / "paired_geometry_results.csv", rows)
    payload = {
        "num_paired_model_pairs": len(rows),
        "contrast": "acceptance(K8V4) - acceptance(K4V8)",
        "inference_unit": "model_pair",
        "naive_macro": naive_macro,
        "kivi_macro": kivi_macro,
        "geometry_shift_kivi_minus_naive": shift_macro,
        "naive_k8v4_win_count": sum(float(row["naive_k8v4_minus_k4v8"]) > 0 for row in rows),
        "kivi_k8v4_win_count": sum(float(row["kivi_k8v4_minus_k4v8"]) > 0 for row in rows),
        "rows": rows,
        "plots": make_plot(rows, args.out_dir),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(args.out_dir / "summary.json")


if __name__ == "__main__":
    main()
