#!/usr/bin/env python3
"""Aggregate multi-seed long-context passkey KV-quantization evaluations."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from spec_kv_statistics import bootstrap_mean_ci


EVALUATOR_VERSION = "kv_multiple_choice_cached_v2"
GENERATOR_VERSION = "synthetic_passkey_v1"


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


def parse_int_list(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def paired_accuracy_differences(
    rows: Iterable[Dict[str, str]],
    *,
    config: str,
    baseline: str,
    context: int,
    depth: float | None = None,
) -> List[float]:
    by_example: Dict[tuple[str, str], Dict[str, float]] = defaultdict(dict)
    for row in rows:
        if int(row["prompt_tokens"]) != int(context):
            continue
        if depth is not None and abs(float(row["passkey_depth"]) - depth) > 1e-9:
            continue
        if row["config"] not in {config, baseline}:
            continue
        key = (row["seed"], row["source_idx"])
        by_example[key][row["config"]] = float(row["raw_correct"])
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
    all_depth = [row for row in rows if row["depth"] == "all"]
    configs = sorted(
        {str(row["config"]) for row in all_depth},
        key=lambda name: (name != "none", name),
    )
    fig, axis = plt.subplots(figsize=(7.6, 4.8))
    for config in configs:
        subset = sorted(
            (row for row in all_depth if row["config"] == config),
            key=lambda row: int(row["context"]),
        )
        axis.plot(
            [int(row["context"]) for row in subset],
            [100.0 * float(row["accuracy_mean"]) for row in subset],
            marker="o",
            label=config.upper() if config != "none" else "BF16",
        )
    axis.set_xscale("log", base=2)
    axis.set_xticks(
        sorted({int(row["context"]) for row in all_depth}),
        [f"{int(row['context']) // 1024}K" for row in sorted(
            {int(item["context"]): item for item in all_depth}.values(),
            key=lambda item: int(item["context"]),
        )],
    )
    axis.set_xlabel("Prefix length")
    axis.set_ylabel("Passkey retrieval accuracy (%)")
    axis.set_ylim(0.0, 103.0)
    axis.grid(alpha=0.22)
    axis.legend(ncol=3, frameon=False)
    axis.set_title("Long-Context Retrieval under KV Quantization", fontweight="bold")
    fig.tight_layout()
    paths = []
    for extension in ("png", "pdf"):
        path = out_dir / f"kv_passkey_accuracy.{extension}"
        fig.savefig(path, dpi=240, bbox_inches="tight")
        paths.append(str(path))
    plt.close(fig)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--expected_contexts", default="4096,8192,16384")
    parser.add_argument("--expected_seeds", default="0,1,2")
    parser.add_argument("--expected_examples_per_run", type=int, default=0)
    parser.add_argument("--require_complete", action="store_true")
    parser.add_argument("--bootstrap_samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    expected_contexts = parse_int_list(args.expected_contexts)
    expected_seeds = parse_int_list(args.expected_seeds)
    all_rows: List[Dict[str, str]] = []
    runs: List[Dict[str, Any]] = []
    missing = []
    underfilled = []
    for context in expected_contexts:
        for seed in expected_seeds:
            seed_dir = args.root / f"ctx_{context}" / f"seed_{seed}"
            summary_path = seed_dir / "summary.json"
            rows_path = seed_dir / "example_rows.csv"
            if not summary_path.exists() or not rows_path.exists():
                missing.append(str(seed_dir))
                continue
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            runtime = summary.get("runtime", {})
            if runtime.get("evaluator_version") != EVALUATOR_VERSION:
                raise ValueError(f"Unexpected evaluator in {summary_path}")
            if runtime.get("task_generator_version") != GENERATOR_VERSION:
                raise ValueError(f"Unexpected passkey generator in {summary_path}")
            if summary.get("task") != "passkey":
                raise ValueError(f"Unexpected task in {summary_path}")
            if int(summary["config"]["max_prompt_tokens"]) != context:
                raise ValueError(f"Context mismatch in {summary_path}")
            if (
                args.expected_examples_per_run > 0
                and int(summary.get("num_examples", -1))
                != args.expected_examples_per_run
            ):
                underfilled.append(
                    {
                        "path": str(summary_path),
                        "observed": int(summary.get("num_examples", -1)),
                        "expected": args.expected_examples_per_run,
                    }
                )
                continue
            all_rows.extend(read_csv(rows_path))
            runs.append(
                {
                    "context": context,
                    "seed": seed,
                    "summary": summary,
                }
            )
    if not runs:
        raise ValueError("No complete passkey runs were found.")
    if args.require_complete and (missing or underfilled):
        raise ValueError(
            f"Passkey sweep is incomplete: missing={missing}, underfilled={underfilled}"
        )

    grouped: List[Dict[str, Any]] = []
    comparisons: List[Dict[str, Any]] = []
    depths: List[float | None] = [None, 0.1, 0.5, 0.9]
    for context in expected_contexts:
        context_runs = [run for run in runs if run["context"] == context]
        if not context_runs:
            continue
        configs = sorted(
            context_runs[0]["summary"]["summaries"],
            key=lambda name: (name != "none", name),
        )
        for depth in depths:
            depth_rows = [
                row
                for row in all_rows
                if int(row["prompt_tokens"]) == context
                and (depth is None or abs(float(row["passkey_depth"]) - depth) <= 1e-9)
            ]
            for config in configs:
                config_rows = [row for row in depth_rows if row["config"] == config]
                if not config_rows:
                    continue
                accuracy = bootstrap_mean_ci(
                    [float(row["raw_correct"]) for row in config_rows],
                    seed=args.seed + len(grouped),
                    samples=args.bootstrap_samples,
                )
                delta_values = paired_accuracy_differences(
                    all_rows,
                    config=config,
                    baseline="none",
                    context=context,
                    depth=depth,
                )
                delta = bootstrap_mean_ci(
                    delta_values,
                    seed=args.seed + 1000 + len(grouped),
                    samples=args.bootstrap_samples,
                )
                summaries = [run["summary"]["summaries"][config] for run in context_runs]
                grouped.append(
                    {
                        "context": context,
                        "depth": "all" if depth is None else depth,
                        "config": config,
                        "num_seeds": len(context_runs),
                        "num_examples": len(config_rows),
                        "accuracy_mean": accuracy["mean"],
                        "accuracy_ci_low": accuracy["ci_low"],
                        "accuracy_ci_high": accuracy["ci_high"],
                        "paired_delta_vs_bf16_mean": delta["mean"],
                        "paired_delta_vs_bf16_ci_low": delta["ci_low"],
                        "paired_delta_vs_bf16_ci_high": delta["ci_high"],
                        "cache_saved_fraction": sum(
                            float(summary["cache_saved_fraction"])
                            for summary in summaries
                        )
                        / len(summaries),
                    }
                )
            for left, right in (
                ("k8v4", "k4v8"),
                ("k4v3", "k3v4"),
                ("k4v2", "k2v4"),
            ):
                if left not in configs or right not in configs:
                    continue
                values = paired_accuracy_differences(
                    all_rows,
                    config=left,
                    baseline=right,
                    context=context,
                    depth=depth,
                )
                contrast = bootstrap_mean_ci(
                    values,
                    seed=args.seed + 2000 + len(comparisons),
                    samples=args.bootstrap_samples,
                )
                comparisons.append(
                    {
                        "context": context,
                        "depth": "all" if depth is None else depth,
                        "config_a": left,
                        "config_b": right,
                        "paired_count": len(values),
                        "accuracy_a_minus_b_mean": contrast["mean"],
                        "accuracy_a_minus_b_ci_low": contrast["ci_low"],
                        "accuracy_a_minus_b_ci_high": contrast["ci_high"],
                    }
                )

    write_csv(args.out_dir / "grouped_results.csv", grouped)
    write_csv(args.out_dir / "paired_comparisons.csv", comparisons)
    payload = {
        "evaluator_version": EVALUATOR_VERSION,
        "task_generator_version": GENERATOR_VERSION,
        "num_complete_runs": len(runs),
        "expected_contexts": expected_contexts,
        "expected_seeds": expected_seeds,
        "expected_examples_per_run": args.expected_examples_per_run,
        "missing_runs": missing,
        "underfilled_runs": underfilled,
        "complete_run_gate": args.require_complete and not missing and not underfilled,
        "grouped": grouped,
        "comparisons": comparisons,
        "plots": make_plot(grouped, args.out_dir),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Aggregated {len(runs)} passkey runs")
    print(args.out_dir / "summary.json")


if __name__ == "__main__":
    main()
