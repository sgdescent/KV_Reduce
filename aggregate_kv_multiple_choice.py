#!/usr/bin/env python3
"""Aggregate multi-seed KV-quantized multiple-choice evaluations."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from spec_kv_statistics import bootstrap_mean_ci


EVALUATOR_VERSION = "kv_multiple_choice_cached_v1"


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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


def paired_metric_differences(
    rows: Iterable[Dict[str, str]],
    *,
    config: str,
    baseline: str,
    metric: str,
) -> List[float]:
    by_example: Dict[tuple[str, str, str], Dict[str, float]] = defaultdict(dict)
    for row in rows:
        key = (row["task"], row["seed"], row["source_idx"])
        if row["config"] in {config, baseline}:
            by_example[key][row["config"]] = float(row[metric])
    return [
        values[config] - values[baseline]
        for values in by_example.values()
        if config in values and baseline in values
    ]


def make_plot(rows: Sequence[Dict[str, Any]], out_dir: Path) -> List[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    tasks = sorted({str(row["task"]) for row in rows})
    fig, axes = plt.subplots(1, len(tasks), figsize=(5.3 * len(tasks), 4.4), squeeze=False)
    for axis, task in zip(axes[0], tasks):
        subset = [row for row in rows if row["task"] == task]
        labels = [str(row["config"]).upper() for row in subset]
        values = [100.0 * float(row["primary_accuracy_mean"]) for row in subset]
        axis.bar(labels, values, color="#2B6F77")
        axis.set_title(task.replace("_", " ").title())
        axis.set_ylabel("Task accuracy (%)")
        axis.tick_params(axis="x", rotation=35)
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("Ordinary-LM Accuracy with KV Quantization", fontweight="bold")
    fig.tight_layout()
    paths = []
    for extension in ("png", "pdf"):
        path = out_dir / f"kv_multiple_choice_accuracy.{extension}"
        fig.savefig(path, dpi=240, bbox_inches="tight")
        paths.append(str(path))
    plt.close(fig)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--bootstrap_samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, str]] = []
    runs: List[Dict[str, Any]] = []
    missing: List[str] = []
    for seed_dir in sorted(args.root.glob("*/seed_*")):
        summary_path = seed_dir / "summary.json"
        rows_path = seed_dir / "example_rows.csv"
        if not summary_path.exists() or not rows_path.exists():
            missing.append(str(seed_dir))
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        version = summary.get("runtime", {}).get("evaluator_version")
        if version != EVALUATOR_VERSION:
            raise ValueError(f"Unexpected evaluator {version!r} in {summary_path}")
        all_rows.extend(read_csv(rows_path))
        runs.append(
            {
                "task": summary["task"],
                "seed": int(summary["config"]["seed"]),
                "num_examples": int(summary["num_examples"]),
                "primary_metric": summary["primary_metric"],
                "summary": summary,
            }
        )
    if not runs:
        raise ValueError("No complete multiple-choice runs were found.")

    grouped: List[Dict[str, Any]] = []
    comparisons: List[Dict[str, Any]] = []
    for task in sorted({str(run["task"]) for run in runs}):
        task_runs = [run for run in runs if run["task"] == task]
        primary = str(task_runs[0]["primary_metric"])
        metric = "normalized_correct" if primary == "normalized_accuracy" else "raw_correct"
        task_rows = [row for row in all_rows if row["task"] == task]
        configs = sorted({row["config"] for row in task_rows}, key=lambda name: (name != "none", name))
        for config_idx, config in enumerate(configs):
            config_rows = [row for row in task_rows if row["config"] == config]
            accuracy = bootstrap_mean_ci(
                [float(row[metric]) for row in config_rows],
                seed=args.seed + len(grouped),
                samples=args.bootstrap_samples,
            )
            delta_values = paired_metric_differences(
                task_rows,
                config=config,
                baseline="none",
                metric=metric,
            )
            delta = bootstrap_mean_ci(
                delta_values,
                seed=args.seed + 1000 + len(grouped),
                samples=args.bootstrap_samples,
            )
            summaries = [run["summary"]["summaries"][config] for run in task_runs]
            grouped.append(
                {
                    "task": task,
                    "primary_metric": primary,
                    "config": config,
                    "num_seeds": len(task_runs),
                    "num_examples": len(config_rows),
                    "primary_accuracy_mean": accuracy["mean"],
                    "primary_accuracy_ci_low": accuracy["ci_low"],
                    "primary_accuracy_ci_high": accuracy["ci_high"],
                    "paired_delta_vs_bf16_mean": delta["mean"],
                    "paired_delta_vs_bf16_ci_low": delta["ci_low"],
                    "paired_delta_vs_bf16_ci_high": delta["ci_high"],
                    "raw_accuracy": sum(float(row["raw_correct"]) for row in config_rows) / len(config_rows),
                    "normalized_accuracy": sum(float(row["normalized_correct"]) for row in config_rows) / len(config_rows),
                    "cache_saved_fraction": sum(float(row["cache_saved_fraction"]) for row in summaries) / len(summaries),
                }
            )

        for left, right in (("k8v4", "k4v8"), ("k4v3", "k3v4")):
            if left not in configs or right not in configs:
                continue
            differences = paired_metric_differences(
                task_rows,
                config=left,
                baseline=right,
                metric=metric,
            )
            contrast = bootstrap_mean_ci(
                differences,
                seed=args.seed + 2000 + len(comparisons),
                samples=args.bootstrap_samples,
            )
            comparisons.append(
                {
                    "task": task,
                    "primary_metric": primary,
                    "config_a": left,
                    "config_b": right,
                    "paired_count": len(differences),
                    "accuracy_a_minus_b_mean": contrast["mean"],
                    "accuracy_a_minus_b_ci_low": contrast["ci_low"],
                    "accuracy_a_minus_b_ci_high": contrast["ci_high"],
                }
            )

    write_csv(args.out_dir / "grouped_results.csv", grouped)
    write_csv(args.out_dir / "paired_comparisons.csv", comparisons)
    payload = {
        "evaluator_version": EVALUATOR_VERSION,
        "num_complete_runs": len(runs),
        "missing_runs": missing,
        "grouped": grouped,
        "comparisons": comparisons,
        "plots": make_plot(grouped, args.out_dir),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Aggregated {len(runs)} multiple-choice runs")
    print(args.out_dir / "summary.json")


if __name__ == "__main__":
    main()
