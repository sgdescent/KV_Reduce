#!/usr/bin/env python3
"""Aggregate multi-budget, multi-context, multi-seed objective KV results."""

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List, Tuple


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


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


def bootstrap_mean_ci(values: List[float], *, seed: int, samples: int = 2000) -> Tuple[float, float, float]:
    if not values:
        return float("nan"), float("nan"), float("nan")
    if len(values) == 1:
        return values[0], values[0], values[0]
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        estimates.append(mean(values[rng.randrange(len(values))] for _ in values))
    estimates.sort()
    return mean(values), estimates[int(0.025 * samples)], estimates[min(samples - 1, int(0.975 * samples))]


def classify_exactness(row: Dict[str, str], *, tie_margin: float) -> str:
    """Classify target-output agreement without hiding finite-precision ties."""
    if float(row["matches_target_greedy"]) >= 0.5:
        return "exact"
    try:
        margin = float(row["mismatch_min_top1_margin"])
    except (KeyError, TypeError, ValueError):
        margin = float("nan")
    if math.isfinite(margin) and margin <= tie_margin:
        return "numerical_tie"
    return "non_tie_or_unknown"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate the objective-aware KV matrix.")
    parser.add_argument("--matrix_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument(
        "--exactness_tie_margin",
        type=float,
        default=1e-3,
        help="Maximum target top-1 margin treated as a finite-precision numerical tie.",
    )
    return parser


def make_plot(grouped_rows: List[Dict[str, Any]], out_dir: Path) -> List[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    contexts = sorted({int(row["context"]) for row in grouped_rows})
    budgets = sorted({int(row["budget"]) for row in grouped_rows})
    fig, axes = plt.subplots(1, len(contexts), figsize=(5.0 * len(contexts), 4.3), squeeze=False)
    colors = {
        "quality": "#26456E",
        "acceptance": "#D1495B",
        "k_priority": "#2A9D8F",
        "v_priority": "#E9C46A",
    }
    objective_order = [
        objective
        for objective in ("quality", "acceptance", "k_priority", "v_priority")
        if any(row["allocation_objective"] == objective for row in grouped_rows)
    ]
    for axis, context in zip(axes[0], contexts):
        for objective in objective_order:
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
    rejected = []
    prompt_effects: Dict[Tuple[int, int], Dict[str, List[float]]] = defaultdict(
        lambda: {"acceptance": [], "quality_kl": [], "quality_delta_nll": []}
    )
    acceptance_prompt_counts: Dict[Tuple[int, int], Dict[str, int]] = defaultdict(
        lambda: {"candidate_pairs": 0, "excluded_non_tie": 0, "used": 0}
    )
    exactness_counts: Dict[Tuple[int, int, int], Dict[str, int]] = defaultdict(
        lambda: {"exact": 0, "numerical_tie": 0, "non_tie_or_unknown": 0, "invalid_prompts": 0}
    )
    exactness_examples: List[Dict[str, Any]] = []

    for budget_dir in sorted(matrix_dir.glob("budget_*")):
        budget = int(budget_dir.name.split("_", 1)[1])
        quality_allocation = read_json(budget_dir / "quality_allocation" / "allocation.json")
        acceptance_allocation = read_json(budget_dir / "acceptance_allocation" / "allocation.json")
        allocations = [("quality", quality_allocation), ("acceptance", acceptance_allocation)]
        for objective in ("k_priority", "v_priority"):
            allocation_path = budget_dir / f"{objective}_allocation" / "allocation.json"
            if allocation_path.exists():
                allocations.append((objective, read_json(allocation_path)))
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
                quality_version = quality_eval.get("runtime", {}).get("evaluator_version")
                acceptance_version = acceptance_eval.get("runtime", {}).get("evaluator_version")
                if quality_version != "teacher_forced_cached_v1" or acceptance_version != "cached_dynamic_v4":
                    rejected.append(
                        {
                            "path": str(seed_dir),
                            "quality_version": quality_version,
                            "acceptance_version": acceptance_version,
                        }
                    )
                    continue
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

                quality_name = str(quality_allocation["name"])
                acceptance_name = str(acceptance_allocation["name"])
                acceptance_rows = read_csv(seed_dir / "acceptance" / "benchmark_rows.csv")
                acceptance_by_prompt: Dict[str, Dict[str, float]] = defaultdict(dict)
                invalid_prompts = set()
                for row in acceptance_rows:
                    status = classify_exactness(row, tie_margin=args.exactness_tie_margin)
                    exactness_counts[(budget, context, seed)][status] += 1
                    if status == "non_tie_or_unknown":
                        invalid_prompts.add(row["prompt_idx"])
                        if len(exactness_examples) < 100:
                            exactness_examples.append(
                                {
                                    "budget": budget,
                                    "context": context,
                                    "seed": seed,
                                    "prompt_idx": int(row["prompt_idx"]),
                                    "config": row["config"],
                                    "mismatch_source": row.get("mismatch_source", ""),
                                    "mismatch_min_top1_margin": row.get("mismatch_min_top1_margin", "nan"),
                                }
                            )
                    if row["config"] in {quality_name, acceptance_name}:
                        acceptance_by_prompt[row["prompt_idx"]][row["config"]] = float(row["accept_rate"])
                exactness_counts[(budget, context, seed)]["invalid_prompts"] = len(invalid_prompts)
                prompt_count = acceptance_prompt_counts[(budget, context)]
                for prompt_idx, pair in acceptance_by_prompt.items():
                    if set(pair) == {quality_name, acceptance_name}:
                        prompt_count["candidate_pairs"] += 1
                        if prompt_idx in invalid_prompts:
                            prompt_count["excluded_non_tie"] += 1
                            continue
                        prompt_count["used"] += 1
                        prompt_effects[(budget, context)]["acceptance"].append(
                            pair[acceptance_name] - pair[quality_name]
                        )

                quality_rows = read_csv(seed_dir / "quality" / "raw_sequence_rows.csv")
                quality_by_sequence: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(dict)
                for row in quality_rows:
                    if row["candidate"] in {quality_name, acceptance_name}:
                        quality_by_sequence[row["sequence_idx"]][row["candidate"]] = {
                            "kl": float(row["kl_p_to_q"]),
                            "delta_nll": float(row["delta_nll"]),
                        }
                for pair in quality_by_sequence.values():
                    if set(pair) == {quality_name, acceptance_name}:
                        prompt_effects[(budget, context)]["quality_kl"].append(
                            pair[acceptance_name]["kl"] - pair[quality_name]["kl"]
                        )
                        prompt_effects[(budget, context)]["quality_delta_nll"].append(
                            pair[acceptance_name]["delta_nll"] - pair[quality_name]["delta_nll"]
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
        if not {"quality", "acceptance"}.issubset(pair):
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
        row = {
                "budget": budget,
                "context": context,
                "num_seeds": len(values),
                "acceptance_optimized_acceptance_advantage_mean": mean(acceptance_advantage),
                "acceptance_optimized_acceptance_advantage_ci95": ci95(acceptance_advantage),
                "quality_optimized_delta_nll_advantage_mean": mean(quality_advantage),
                "quality_optimized_delta_nll_advantage_ci95": ci95(quality_advantage),
            }
        paired = prompt_effects[(budget, context)]
        for metric, metric_values in paired.items():
            estimate, low, high = bootstrap_mean_ci(
                metric_values,
                seed=budget * 100000 + context * 10 + len(metric),
            )
            row[f"paired_{metric}_n"] = len(metric_values)
            row[f"paired_{metric}_mean"] = estimate
            row[f"paired_{metric}_ci_low"] = low
            row[f"paired_{metric}_ci_high"] = high
        prompt_count = acceptance_prompt_counts[(budget, context)]
        row["paired_acceptance_candidate_n"] = prompt_count["candidate_pairs"]
        row["paired_acceptance_excluded_non_tie_n"] = prompt_count["excluded_non_tie"]
        row["paired_acceptance_valid_n"] = prompt_count["used"]
        effect_rows.append(row)

    budget_prompt_effects: Dict[int, Dict[str, List[float]]] = defaultdict(
        lambda: {"acceptance": [], "quality_kl": [], "quality_delta_nll": []}
    )
    for (budget, _), metrics in prompt_effects.items():
        for metric, values in metrics.items():
            budget_prompt_effects[budget][metric].extend(values)
    cross_context_rows = []
    for budget, metrics in sorted(budget_prompt_effects.items()):
        row: Dict[str, Any] = {"budget": budget}
        for metric, values in metrics.items():
            estimate, low, high = bootstrap_mean_ci(
                values,
                seed=budget * 1000000 + len(metric),
                samples=5000,
            )
            row[f"paired_{metric}_n"] = len(values)
            row[f"paired_{metric}_mean"] = estimate
            row[f"paired_{metric}_ci_low"] = low
            row[f"paired_{metric}_ci_high"] = high
        cross_context_rows.append(row)

    exactness_rows = []
    for (budget, context, seed), counts in sorted(exactness_counts.items()):
        exactness_rows.append({"budget": budget, "context": context, "seed": seed, **counts})
    exactness_totals = {
        key: sum(row[key] for row in exactness_rows)
        for key in ("exact", "numerical_tie", "non_tie_or_unknown", "invalid_prompts")
    }

    write_csv(rows, out_dir / "matrix_rows.csv")
    write_csv(grouped_rows, out_dir / "matrix_grouped.csv")
    write_csv(effect_rows, out_dir / "cross_objective_effects.csv")
    write_csv(cross_context_rows, out_dir / "cross_context_effects.csv")
    write_csv(exactness_rows, out_dir / "exactness_audit.csv")
    payload = {
        "num_complete_rows": len(rows),
        "num_missing_pairs": len(missing),
        "num_rejected_pairs": len(rejected),
        "missing_pairs": missing,
        "rejected_pairs": rejected,
        "exactness_tie_margin": args.exactness_tie_margin,
        "exactness_audit": {
            "totals": exactness_totals,
            "cells": exactness_rows,
            "non_tie_or_unknown_examples": exactness_examples,
        },
        "grouped": grouped_rows,
        "cross_objective_effects": effect_rows,
        "cross_context_effects": cross_context_rows,
        "plots": make_plot(grouped_rows, out_dir),
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Aggregated {len(rows)} rows; missing pairs: {len(missing)}")


if __name__ == "__main__":
    main()
