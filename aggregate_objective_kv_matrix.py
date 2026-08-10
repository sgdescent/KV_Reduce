#!/usr/bin/env python3
"""Aggregate multi-budget, multi-context, multi-seed objective KV results."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List, Tuple


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def ci95(values: List[float]) -> float:
    return 1.96 * stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate the objective-aware KV matrix.")
    parser.add_argument("--matrix_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    return parser


def make_plot(grouped_rows: List[Dict[str, Any]], out_dir: Path) -> List[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    contexts = sorted({int(row["context"]) for row in grouped_rows})
    budgets = sorted({int(row["budget"]) for row in grouped_rows})
    fig, axes = plt.subplots(1, len(contexts), figsize=(5.0 * len(contexts), 4.3), squeeze=False)
    colors = {"quality": "#26456E", "acceptance": "#D1495B"}
    for axis, context in zip(axes[0], contexts):
        for objective in ("quality", "acceptance"):
            subset = sorted(
                [row for row in grouped_rows if int(row["context"]) == context and row["allocation_objective"] == objective],
                key=lambda row: int(row["budget"]),
            )
            axis.errorbar(
                [int(row["budget"]) for row in subset],
                [float(row["spec_accept_rate_mean"]) for row in subset],
                yerr=[float(row["spec_accept_rate_ci95"]) for row in subset],
                marker="o",
                linewidth=2,
                color=colors[objective],
                label=f"{objective}-optimized",
            )
        axis.set_title(f"Context {context}")
        axis.set_xlabel("Profiled mean KV bits")
        axis.set_ylabel("Speculative acceptance")
        axis.grid(alpha=0.25)
    axes[0][0].legend()
    fig.suptitle("Objective-Aware KV Allocation Across Context and Memory", fontweight="bold")
    fig.tight_layout()
    paths = []
    for extension in ("png", "pdf"):
        path = out_dir / f"objective_matrix_acceptance.{extension}"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        paths.append(str(path))
    plt.close(fig)
    return paths


def main() -> None:
    args = build_parser().parse_args()
    matrix_dir = Path(args.matrix_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    missing = []

    for budget_dir in sorted(matrix_dir.glob("budget_*")):
        budget = int(budget_dir.name.split("_", 1)[1])
        quality_allocation = read_json(budget_dir / "quality_allocation" / "allocation.json")
        acceptance_allocation = read_json(budget_dir / "acceptance_allocation" / "allocation.json")
        allocations = (("quality", quality_allocation), ("acceptance", acceptance_allocation))
        for context_dir in sorted(budget_dir.glob("ctx_*")):
            context = int(context_dir.name.split("_", 1)[1])
            for seed_dir in sorted(context_dir.glob("seed_*")):
                seed = int(seed_dir.name.split("_", 1)[1])
                quality_path = seed_dir / "quality" / "summary.json"
                acceptance_path = seed_dir / "acceptance" / "summary.json"
                if not quality_path.exists() or not acceptance_path.exists():
                    missing.append(str(seed_dir))
                    continue
                quality_eval = read_json(quality_path)
                acceptance_eval = read_json(acceptance_path)
                for allocation_objective, allocation in allocations:
                    name = str(allocation["name"])
                    quality = quality_eval["summaries"][name]
                    acceptance = acceptance_eval["summaries"][name]
                    rows.append(
                        {
                            "budget": budget,
                            "context": context,
                            "seed": seed,
                            "allocation_objective": allocation_objective,
                            "allocation_name": name,
                            "profiled_mean_bits": allocation["achieved_profiled_mean_bits"],
                            "all_component_mean_bits": quality["allocation/all_bits_mean"],
                            "quality_delta_nll": quality["delta_nll"],
                            "quality_kl": quality["kl_p_to_q"],
                            "quality_top1_match": quality["top1_match"],
                            "spec_accept_rate": acceptance["overall_accept_rate"],
                            "spec_accepted_per_round": acceptance["accepted_per_round"],
                            "spec_round_js": acceptance["round_js"],
                            "draft_cache_saved_fraction": acceptance["draft_cache_saved_fraction"],
                            "total_cache_saved_fraction": acceptance["total_cache_saved_fraction"],
                        }
                    )

    if not rows:
        raise ValueError("No complete matrix result pairs were found.")

    groups: Dict[Tuple[int, int, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(int(row["budget"]), int(row["context"]), str(row["allocation_objective"]))].append(row)
    grouped_rows: List[Dict[str, Any]] = []
    for (budget, context, objective), group in sorted(groups.items()):
        accept = [float(row["spec_accept_rate"]) for row in group]
        delta_nll = [float(row["quality_delta_nll"]) for row in group]
        quality_kl = [float(row["quality_kl"]) for row in group]
        grouped_rows.append(
            {
                "budget": budget,
                "context": context,
                "allocation_objective": objective,
                "num_seeds": len(group),
                "spec_accept_rate_mean": mean(accept),
                "spec_accept_rate_ci95": ci95(accept),
                "quality_delta_nll_mean": mean(delta_nll),
                "quality_delta_nll_ci95": ci95(delta_nll),
                "quality_kl_mean": mean(quality_kl),
                "quality_kl_ci95": ci95(quality_kl),
                "total_cache_saved_fraction": mean(float(row["total_cache_saved_fraction"]) for row in group),
            }
        )

    paired: Dict[Tuple[int, int, int], Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        paired[(int(row["budget"]), int(row["context"]), int(row["seed"]))][str(row["allocation_objective"])] = row
    effects: Dict[Tuple[int, int], List[Tuple[float, float]]] = defaultdict(list)
    for (budget, context, _), pair in paired.items():
        if set(pair) != {"quality", "acceptance"}:
            continue
        effects[(budget, context)].append(
            (
                float(pair["acceptance"]["spec_accept_rate"]) - float(pair["quality"]["spec_accept_rate"]),
                float(pair["acceptance"]["quality_delta_nll"]) - float(pair["quality"]["quality_delta_nll"]),
            )
        )
    effect_rows = []
    for (budget, context), values in sorted(effects.items()):
        acceptance_advantage = [value[0] for value in values]
        quality_advantage = [value[1] for value in values]
        effect_rows.append(
            {
                "budget": budget,
                "context": context,
                "num_seeds": len(values),
                "acceptance_optimized_acceptance_advantage_mean": mean(acceptance_advantage),
                "acceptance_optimized_acceptance_advantage_ci95": ci95(acceptance_advantage),
                "quality_optimized_delta_nll_advantage_mean": mean(quality_advantage),
                "quality_optimized_delta_nll_advantage_ci95": ci95(quality_advantage),
            }
        )

    write_csv(rows, out_dir / "matrix_rows.csv")
    write_csv(grouped_rows, out_dir / "matrix_grouped.csv")
    write_csv(effect_rows, out_dir / "cross_objective_effects.csv")
    payload = {
        "num_complete_rows": len(rows),
        "num_missing_pairs": len(missing),
        "missing_pairs": missing,
        "grouped": grouped_rows,
        "cross_objective_effects": effect_rows,
        "plots": make_plot(grouped_rows, out_dir),
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Aggregated {len(rows)} rows; missing pairs: {len(missing)}")


if __name__ == "__main__":
    main()
