#!/usr/bin/env python3
"""Aggregate a multi-budget, multi-context speculative draft-length ablation."""

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List, Tuple

from aggregate_objective_kv_matrix import classify_exactness


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def bootstrap_mean_ci(values: List[float], *, seed: int, samples: int = 3000) -> Tuple[float, float, float]:
    if not values:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    estimates = [mean(values[rng.randrange(len(values))] for _ in values) for _ in range(samples)]
    estimates.sort()
    return mean(values), estimates[int(0.025 * samples)], estimates[min(samples - 1, int(0.975 * samples))]


def make_plot(grouped: List[Dict[str, Any]], out_dir: Path) -> List[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    contexts = sorted({int(row["context"]) for row in grouped})
    budgets = sorted({int(row["budget"]) for row in grouped})
    configs = ["none", "quality", "acceptance", "k_priority", "v_priority"]
    colors = {
        "none": "#555555",
        "quality": "#26456E",
        "acceptance": "#D1495B",
        "k_priority": "#2A9D8F",
        "v_priority": "#E9C46A",
    }
    fig, axes = plt.subplots(len(budgets), len(contexts), figsize=(5 * len(contexts), 3.8 * len(budgets)), squeeze=False)
    for row_idx, budget in enumerate(budgets):
        for col_idx, context in enumerate(contexts):
            axis = axes[row_idx][col_idx]
            for config in configs:
                subset = sorted(
                    [
                        row
                        for row in grouped
                        if int(row["budget"]) == budget
                        and int(row["context"]) == context
                        and row["config_role"] == config
                    ],
                    key=lambda row: int(row["draft_steps"]),
                )
                if not subset:
                    continue
                axis.errorbar(
                    [row["draft_steps"] for row in subset],
                    [row["accept_rate_mean"] for row in subset],
                    yerr=[row["accept_rate_ci95"] for row in subset],
                    marker="o",
                    label=config,
                    color=colors[config],
                )
            axis.set_title(f"{budget}-bit profile, context {context}")
            axis.set_xlabel("Draft proposals per round")
            axis.set_ylabel("Acceptance rate")
            axis.grid(alpha=0.25)
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    paths = []
    for extension in ("png", "pdf"):
        path = out_dir / f"gamma_acceptance.{extension}"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        paths.append(str(path))
    plt.close(fig)
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate the speculative draft-length matrix.")
    parser.add_argument("--matrix_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--exactness_tie_margin", type=float, default=1e-3)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.matrix_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result_rows = []
    missing = []
    rejected = []
    prompt_effects: Dict[Tuple[int, int, int, str], List[float]] = defaultdict(list)
    exactness = {"exact": 0, "numerical_tie": 0, "non_tie_or_unknown": 0, "invalid_prompts": 0}

    for summary_path in sorted(root.glob("budget_*/ctx_*/gamma_*/seed_*/summary.json")):
        budget = int(summary_path.parents[3].name.split("_")[1])
        context = int(summary_path.parents[2].name.split("_")[1])
        draft_steps = int(summary_path.parents[1].name.split("_")[1])
        seed = int(summary_path.parent.name.split("_")[1])
        payload = read_json(summary_path)
        version = payload.get("runtime", {}).get("evaluator_version")
        if version != "cached_dynamic_v4":
            rejected.append({"path": str(summary_path), "evaluator_version": version})
            continue
        role_names = {
            "none": "none",
            "quality": f"quality_b{budget}",
            "acceptance": f"acceptance_b{budget}",
            "k_priority": f"k_priority_b{budget}",
            "v_priority": f"v_priority_b{budget}",
        }
        summaries = payload["summaries"]
        for role, name in role_names.items():
            if name not in summaries:
                missing.append(f"{summary_path}:{name}")
                continue
            summary = summaries[name]
            result_rows.append(
                {
                    "budget": budget,
                    "context": context,
                    "draft_steps": draft_steps,
                    "seed": seed,
                    "config_role": role,
                    "config_name": name,
                    "accept_rate": summary["overall_accept_rate"],
                    "accepted_per_round": summary["accepted_per_round"],
                    "full_accept_round_fraction": summary["full_accept_round_fraction"],
                    "round_js": summary["round_js"],
                    "draft_cache_saved_fraction": summary["draft_cache_saved_fraction"],
                    "total_cache_saved_fraction": summary["total_cache_saved_fraction"],
                }
            )

        benchmark_path = summary_path.parent / "benchmark_rows.csv"
        prompt_rows = read_csv(benchmark_path)
        by_prompt: Dict[str, Dict[str, float]] = defaultdict(dict)
        invalid_prompts = set()
        for row in prompt_rows:
            status = classify_exactness(row, tie_margin=args.exactness_tie_margin)
            exactness[status] += 1
            if status == "non_tie_or_unknown":
                invalid_prompts.add(row["prompt_idx"])
            for role, name in role_names.items():
                if row["config"] == name:
                    by_prompt[row["prompt_idx"]][role] = float(row["accept_rate"])
        exactness["invalid_prompts"] += len(invalid_prompts)
        for prompt_idx, values in by_prompt.items():
            if prompt_idx in invalid_prompts or "acceptance" not in values:
                continue
            for baseline in ("quality", "k_priority", "v_priority"):
                if baseline in values:
                    prompt_effects[(budget, context, draft_steps, baseline)].append(
                        values["acceptance"] - values[baseline]
                    )

    expected_dirs = list(root.glob("budget_*/ctx_*/gamma_*/seed_*"))
    for result_dir in expected_dirs:
        if not (result_dir / "summary.json").exists():
            missing.append(str(result_dir))
    if not result_rows:
        raise ValueError("No valid gamma-ablation summaries were found.")

    grouped_values: Dict[Tuple[int, int, int, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in result_rows:
        grouped_values[(row["budget"], row["context"], row["draft_steps"], row["config_role"])].append(row)
    grouped = []
    for (budget, context, draft_steps, role), values in sorted(grouped_values.items()):
        acceptance = [float(row["accept_rate"]) for row in values]
        grouped.append(
            {
                "budget": budget,
                "context": context,
                "draft_steps": draft_steps,
                "config_role": role,
                "num_seeds": len(values),
                "accept_rate_mean": mean(acceptance),
                "accept_rate_ci95": ci95(acceptance),
                "accepted_per_round_mean": mean(float(row["accepted_per_round"]) for row in values),
                "total_cache_saved_fraction": mean(float(row["total_cache_saved_fraction"]) for row in values),
            }
        )

    effects = []
    for (budget, context, draft_steps, baseline), values in sorted(prompt_effects.items()):
        estimate, low, high = bootstrap_mean_ci(
            values,
            seed=budget * 100000 + context * 10 + draft_steps + len(baseline),
        )
        effects.append(
            {
                "budget": budget,
                "context": context,
                "draft_steps": draft_steps,
                "baseline": baseline,
                "paired_n": len(values),
                "acceptance_allocation_advantage_mean": estimate,
                "acceptance_allocation_advantage_ci_low": low,
                "acceptance_allocation_advantage_ci_high": high,
            }
        )

    write_csv(result_rows, out_dir / "gamma_rows.csv")
    write_csv(grouped, out_dir / "gamma_grouped.csv")
    write_csv(effects, out_dir / "gamma_effects.csv")
    output = {
        "num_valid_rows": len(result_rows),
        "num_missing": len(missing),
        "num_rejected": len(rejected),
        "missing": missing,
        "rejected": rejected,
        "exactness": exactness,
        "grouped": grouped,
        "effects": effects,
        "plots": make_plot(grouped, out_dir),
    }
    (out_dir / "summary.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Aggregated {len(result_rows)} gamma rows; missing={len(missing)} rejected={len(rejected)}")


if __name__ == "__main__":
    main()
