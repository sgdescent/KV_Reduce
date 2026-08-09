#!/usr/bin/env python3
"""Aggregate the equal-memory cross-evaluation into a paper-ready result."""

import argparse
import csv
import json
import os
from typing import Any, Dict, List

from kv_utils import write_json


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(rows: List[Dict[str, Any]], path: str) -> None:
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_plot(rows: List[Dict[str, Any]], out_dir: str) -> List[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    labels = [row["allocation"] for row in rows]
    colors = ["#26456E", "#D1495B"]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2))
    axes[0].bar(labels, [row["quality_delta_nll"] for row in rows], color=colors[: len(rows)])
    axes[0].set_title("Ordinary LM Quality")
    axes[0].set_ylabel("Delta NLL (lower is better)")
    axes[1].bar(labels, [row["spec_accept_rate"] for row in rows], color=colors[: len(rows)])
    axes[1].set_title("Speculative Decoding")
    axes[1].set_ylabel("Acceptance rate (higher is better)")
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=12)
    fig.suptitle("Same KV Memory, Different Downstream Objectives", fontweight="bold")
    fig.tight_layout()
    paths = []
    for extension in ("png", "pdf"):
        path = os.path.join(out_dir, f"objective_cross_evaluation.{extension}")
        fig.savefig(path, dpi=220, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate objective-aware KV cross-evaluation.")
    parser.add_argument("--quality_summary", type=str, required=True)
    parser.add_argument("--acceptance_summary", type=str, required=True)
    parser.add_argument("--quality_allocation", type=str, required=True)
    parser.add_argument("--acceptance_allocation", type=str, required=True)
    parser.add_argument("--comparison_summary", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="outputs/kv_objective_final")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    quality_eval = read_json(args.quality_summary)
    acceptance_eval = read_json(args.acceptance_summary)
    quality_allocation = read_json(args.quality_allocation)
    acceptance_allocation = read_json(args.acceptance_allocation)
    allocations = [quality_allocation, acceptance_allocation]

    quality_summaries = quality_eval["summaries"]
    acceptance_summaries = acceptance_eval["summaries"]
    baseline_quality = quality_summaries["none"]
    baseline_acceptance = acceptance_summaries["none"]
    rows: List[Dict[str, Any]] = []
    for allocation in allocations:
        name = str(allocation["name"])
        quality = quality_summaries[name]
        acceptance = acceptance_summaries[name]
        rows.append(
            {
                "allocation": name,
                "profile_objective": allocation["risk_field"],
                "profiled_mean_bits": allocation["achieved_profiled_mean_bits"],
                "all_component_mean_bits": quality["allocation/all_bits_mean"],
                "quality_delta_nll": quality["delta_nll"],
                "quality_js": quality["js"],
                "quality_top1_match": quality["top1_match"],
                "spec_accept_rate": acceptance["overall_accept_rate"],
                "spec_accept_rate_delta": acceptance["overall_accept_rate"]
                - baseline_acceptance["overall_accept_rate"],
                "spec_accepted_per_round": acceptance["accepted_per_round"],
                "spec_round_js": acceptance["round_js"],
                "draft_cache_saved_fraction": acceptance["draft_cache_saved_fraction"],
                "total_cache_saved_fraction": acceptance["total_cache_saved_fraction"],
            }
        )

    equal_memory = abs(rows[0]["all_component_mean_bits"] - rows[1]["all_component_mean_bits"]) < 1e-9
    by_name = {row["allocation"]: row for row in rows}
    quality_row = by_name[quality_allocation["name"]]
    acceptance_row = by_name[acceptance_allocation["name"]]
    payload: Dict[str, Any] = {
        "evaluator_versions": {
            "quality": quality_eval.get("runtime", {}).get("evaluator_version"),
            "acceptance": acceptance_eval.get("runtime", {}).get("evaluator_version"),
        },
        "equal_memory": equal_memory,
        "baseline": {
            "quality_nll": baseline_quality["quantized_nll"],
            "spec_accept_rate": baseline_acceptance["overall_accept_rate"],
        },
        "rows": rows,
        "cross_objective_effect": {
            "acceptance_optimized_acceptance_advantage": acceptance_row["spec_accept_rate"]
            - quality_row["spec_accept_rate"],
            "quality_optimized_delta_nll_advantage": acceptance_row["quality_delta_nll"]
            - quality_row["quality_delta_nll"],
        },
    }
    if args.comparison_summary:
        payload["sensitivity_comparison"] = read_json(args.comparison_summary)
    payload["plots"] = make_plot(rows, args.out_dir)
    write_csv(rows, os.path.join(args.out_dir, "objective_cross_evaluation.csv"))
    write_json(payload, os.path.join(args.out_dir, "summary.json"))
    print("Done!")
    print(f"  equal_memory={equal_memory}")
    print(
        "  acceptance advantage="
        f"{payload['cross_objective_effect']['acceptance_optimized_acceptance_advantage']:+.5f}"
    )
    print(
        "  quality delta-NLL advantage="
        f"{payload['cross_objective_effect']['quality_optimized_delta_nll_advantage']:+.6f}"
    )


if __name__ == "__main__":
    main()
