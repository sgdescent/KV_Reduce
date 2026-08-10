#!/usr/bin/env python3
"""Aggregate fixed K/V precision across speculative proposal lengths."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from aggregate_value_precision_sweep import (
    aggregate_prompt_effects,
    bootstrap_mean_ci,
    parse_config_bits,
    read_csv,
    read_json,
    write_csv,
)


def make_plot(rows: List[Dict[str, Any]], out_dir: Path) -> List[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    fig, axis = plt.subplots(figsize=(6.2, 4.4))
    for config in sorted({str(row["config"]) for row in rows}):
        values = sorted(
            (row for row in rows if row["config"] == config),
            key=lambda row: int(row["draft_steps"]),
        )
        axis.errorbar(
            [int(row["draft_steps"]) for row in values],
            [100.0 * float(row["paired_acceptance_delta_mean"]) for row in values],
            yerr=[
                [
                    100.0
                    * (
                        float(row["paired_acceptance_delta_mean"])
                        - float(row["paired_acceptance_delta_ci_low"])
                    )
                    for row in values
                ],
                [
                    100.0
                    * (
                        float(row["paired_acceptance_delta_ci_high"])
                        - float(row["paired_acceptance_delta_mean"])
                    )
                    for row in values
                ],
            ],
            marker="o",
            capsize=3,
            label=config,
        )
    axis.axhline(0.0, color="#222222", linewidth=1)
    axis.set_xlabel("Draft proposal length (gamma)")
    axis.set_ylabel("Acceptance change vs BF16 (pp)")
    axis.set_xticks(sorted({int(row["draft_steps"]) for row in rows}))
    axis.grid(alpha=0.22)
    axis.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    paths = []
    for extension in ("png", "pdf"):
        path = out_dir / f"value_precision_gamma.{extension}"
        fig.savefig(path, dpi=240, bbox_inches="tight")
        paths.append(str(path))
    plt.close(fig)
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep_dir", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--exactness_tie_margin", type=float, default=1e-3)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_rows: List[Dict[str, Any]] = []
    prompt_effects: Dict[Tuple[int, str], List[float]] = defaultdict(list)
    exactness: Counter[str] = Counter()
    invalid_prompts = 0
    missing = []

    for seed_dir in sorted(args.sweep_dir.glob("gamma_*/seed_*")):
        summary_path = seed_dir / "summary.json"
        benchmark_path = seed_dir / "benchmark_rows.csv"
        if not summary_path.exists() or not benchmark_path.exists():
            missing.append(str(seed_dir))
            continue
        summary = read_json(summary_path)
        version = str(summary.get("runtime", {}).get("evaluator_version", ""))
        if not version.startswith("cached_dynamic_v"):
            raise ValueError(f"Stale evaluator {version!r} in {summary_path}")
        config = summary["config"]
        draft_steps = int(config["draft_steps"])
        seed = int(config["seed"])
        names = [name for name in summary["quant_configs"] if name != "none"]
        effects, counts, invalid = aggregate_prompt_effects(
            read_csv(benchmark_path),
            configs=names,
            tie_margin=args.exactness_tie_margin,
        )
        exactness.update(counts)
        invalid_prompts += invalid
        for name, values in effects.items():
            prompt_effects[(draft_steps, name)].extend(values)
        for name in names:
            metrics = summary["summaries"][name]
            k_bits, v_bits = parse_config_bits(name)
            run_rows.append(
                {
                    "draft_steps": draft_steps,
                    "seed": seed,
                    "config": name,
                    "k_bits": k_bits,
                    "v_bits": v_bits,
                    "accept_rate": metrics["overall_accept_rate"],
                    "accepted_per_round": metrics["accepted_per_round"],
                    "total_cache_saved_fraction": metrics["total_cache_saved_fraction"],
                }
            )

    grouped_values: Dict[Tuple[int, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        grouped_values[(int(row["draft_steps"]), str(row["config"]))].append(row)
    grouped: List[Dict[str, Any]] = []
    for (draft_steps, name), values in sorted(grouped_values.items()):
        effect = bootstrap_mean_ci(
            prompt_effects[(draft_steps, name)],
            seed=draft_steps * 1000 + sum(map(ord, name)),
        )
        grouped.append(
            {
                "draft_steps": draft_steps,
                "config": name,
                "k_bits": values[0]["k_bits"],
                "v_bits": values[0]["v_bits"],
                "num_seeds": len(values),
                "paired_prompt_count": len(prompt_effects[(draft_steps, name)]),
                "paired_acceptance_delta_mean": effect["mean"],
                "paired_acceptance_delta_ci_low": effect["ci_low"],
                "paired_acceptance_delta_ci_high": effect["ci_high"],
                "accepted_per_round_mean": statistics.mean(
                    float(row["accepted_per_round"]) for row in values
                ),
                "total_cache_saved_fraction": statistics.mean(
                    float(row["total_cache_saved_fraction"]) for row in values
                ),
            }
        )
    if not grouped:
        raise ValueError("No complete value-precision gamma outputs were found.")

    write_csv(args.out_dir / "run_results.csv", run_rows)
    write_csv(args.out_dir / "grouped_results.csv", grouped)
    payload = {
        "num_complete_runs": len({(row["draft_steps"], row["seed"]) for row in run_rows}),
        "missing_runs": missing,
        "exactness": dict(exactness),
        "invalid_prompt_occurrences": invalid_prompts,
        "grouped": grouped,
        "plots": make_plot(grouped, args.out_dir),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(args.out_dir / "summary.json")


if __name__ == "__main__":
    main()
