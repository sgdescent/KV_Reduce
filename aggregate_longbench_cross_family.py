#!/usr/bin/env python3
"""Strict cross-family aggregate for LongBench passage retrieval."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


EVALUATOR_VERSION = "kv_multiple_choice_cached_v2"
DATASET_REVISION = "5e628be450b7e67fb7ae6e201bd6d8f7056f7672"
TASK = "longbench_passage_retrieval"
PRIMARY_METRIC = "normalized_accuracy"
CONFIGS = ("none", "k8v4", "k4v8", "k4v4", "k4v2", "k2v4", "k2v2")


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
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


def percentile(sorted_values: Sequence[float], fraction: float) -> float:
    if not sorted_values:
        return float("nan")
    index = min(len(sorted_values) - 1, int(fraction * len(sorted_values)))
    return float(sorted_values[index])


def hierarchical_macro_ci(
    groups: Sequence[Sequence[float]],
    *,
    samples: int,
    seed: int,
) -> Tuple[float, float, float]:
    """Resample models, then paired examples within each sampled model."""

    if not groups or any(not group for group in groups):
        raise ValueError("Hierarchical bootstrap groups must be non-empty.")
    estimate = mean(mean(float(value) for value in group) for group in groups)
    if samples <= 0:
        return estimate, estimate, estimate
    rng = random.Random(seed)
    draws = []
    for _ in range(samples):
        model_means = []
        for _model in groups:
            group = groups[rng.randrange(len(groups))]
            model_means.append(
                mean(float(group[rng.randrange(len(group))]) for _ in group)
            )
        draws.append(mean(model_means))
    draws.sort()
    return estimate, percentile(draws, 0.025), percentile(draws, 0.975)


def pair_effects(
    rows: Iterable[Mapping[str, str]],
    *,
    config_a: str,
    config_b: str,
) -> List[float]:
    paired: Dict[Tuple[str, str], Dict[str, float]] = {}
    for row in rows:
        if row["config"] not in {config_a, config_b}:
            continue
        key = (str(row["seed"]), str(row["source_idx"]))
        paired.setdefault(key, {})[str(row["config"])] = float(
            row["normalized_correct"]
        )
    return [
        values[config_a] - values[config_b]
        for values in paired.values()
        if config_a in values and config_b in values
    ]


def validate_model(
    root: Path,
    model: str,
    *,
    expected_examples: int,
) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    model_root = root / model
    aggregate_path = model_root / "aggregate" / "summary.json"
    if not aggregate_path.exists():
        raise ValueError(f"Missing aggregate for {model}: {aggregate_path}")
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    checks = {
        "evaluator": aggregate.get("evaluator_version") == EVALUATOR_VERSION,
        "complete": aggregate.get("complete_run_gate") is True,
        "runs": int(aggregate.get("num_complete_runs", -1)) == 3,
        "tasks": aggregate.get("expected_tasks") == [TASK],
        "seeds": aggregate.get("expected_seeds") == [0, 1, 2],
        "configs": set(aggregate.get("expected_configs", [])) == set(CONFIGS),
        "revision": aggregate.get("expected_dataset_revision") == DATASET_REVISION,
    }
    failed = [key for key, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"{model} failed aggregate gates: {failed}")

    rows: List[Dict[str, str]] = []
    for seed in range(3):
        path = model_root / TASK / f"seed_{seed}" / "example_rows.csv"
        if not path.exists():
            raise ValueError(f"Missing rows for {model}, seed {seed}: {path}")
        rows.extend(read_csv(path))
    for config in CONFIGS:
        config_rows = [row for row in rows if row["config"] == config]
        if len(config_rows) != expected_examples:
            raise ValueError(
                f"{model}/{config} has {len(config_rows)} rows, expected {expected_examples}."
            )
        ids = {(row["seed"], row["source_idx"]) for row in config_rows}
        if len(ids) != expected_examples:
            raise ValueError(f"{model}/{config} contains duplicate example IDs.")
    return aggregate, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--models", default="qwen25_15b,llama32_3b,qwen3_4b")
    parser.add_argument("--expected_examples_per_model", type=int, default=72)
    parser.add_argument("--bootstrap_samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    if not models:
        raise ValueError("At least one model is required.")

    aggregates: Dict[str, Dict[str, Any]] = {}
    rows_by_model: Dict[str, List[Dict[str, str]]] = {}
    for model in models:
        aggregate, rows = validate_model(
            args.root,
            model,
            expected_examples=args.expected_examples_per_model,
        )
        aggregates[model] = aggregate
        rows_by_model[model] = rows

    model_rows: List[Dict[str, Any]] = []
    macro_rows: List[Dict[str, Any]] = []
    contrasts = (("k8v4", "k4v8"), ("k4v2", "k2v4"))
    for model in models:
        grouped = {
            str(row["config"]): row
            for row in aggregates[model].get("grouped", [])
            if str(row.get("task")) == TASK
        }
        for config in CONFIGS:
            effects = pair_effects(rows_by_model[model], config_a=config, config_b="none")
            model_rows.append(
                {
                    "model": model,
                    "config": config,
                    "num_examples": len(effects),
                    "accuracy": float(grouped[config]["primary_accuracy_mean"]),
                    "delta_vs_bf16": mean(effects),
                    "cache_saved_fraction": float(grouped[config]["cache_saved_fraction"]),
                }
            )
    for config in CONFIGS:
        groups = [
            pair_effects(rows_by_model[model], config_a=config, config_b="none")
            for model in models
        ]
        estimate, low, high = hierarchical_macro_ci(
            groups,
            samples=args.bootstrap_samples,
            seed=args.seed + len(macro_rows),
        )
        macro_rows.append(
            {
                "comparison": f"{config}-none",
                "num_models": len(models),
                "examples_per_model": args.expected_examples_per_model,
                "accuracy_difference": estimate,
                "ci_low": low,
                "ci_high": high,
                "cache_saved_fraction_macro": mean(
                    row["cache_saved_fraction"]
                    for row in model_rows
                    if row["config"] == config
                ),
            }
        )
    for config_a, config_b in contrasts:
        groups = [
            pair_effects(rows_by_model[model], config_a=config_a, config_b=config_b)
            for model in models
        ]
        estimate, low, high = hierarchical_macro_ci(
            groups,
            samples=args.bootstrap_samples,
            seed=args.seed + 100 + len(macro_rows),
        )
        macro_rows.append(
            {
                "comparison": f"{config_a}-{config_b}",
                "num_models": len(models),
                "examples_per_model": args.expected_examples_per_model,
                "accuracy_difference": estimate,
                "ci_low": low,
                "ci_high": high,
                "cache_saved_fraction_macro": mean(
                    row["cache_saved_fraction"]
                    for row in model_rows
                    if row["config"] == config_a
                ),
            }
        )

    depth_rows: List[Dict[str, Any]] = []
    for depth, lower, upper in (
        ("early", 0.0, 1.0 / 3.0),
        ("middle", 1.0 / 3.0, 2.0 / 3.0),
        ("late", 2.0 / 3.0, 1.0),
    ):
        for config_a, config_b in contrasts:
            groups = []
            for model in models:
                subset = [
                    row
                    for row in rows_by_model[model]
                    if lower <= float(row["answer_depth"]) < upper
                    or (depth == "late" and float(row["answer_depth"]) == upper)
                ]
                groups.append(pair_effects(subset, config_a=config_a, config_b=config_b))
            estimate, low, high = hierarchical_macro_ci(
                groups,
                samples=args.bootstrap_samples,
                seed=args.seed + 200 + len(depth_rows),
            )
            depth_rows.append(
                {
                    "depth": depth,
                    "comparison": f"{config_a}-{config_b}",
                    "num_models": len(models),
                    "accuracy_difference": estimate,
                    "ci_low": low,
                    "ci_high": high,
                }
            )

    write_csv(args.out_dir / "model_results.csv", model_rows)
    write_csv(args.out_dir / "macro_results.csv", macro_rows)
    write_csv(args.out_dir / "depth_macro_results.csv", depth_rows)
    payload = {
        "evaluator_version": EVALUATOR_VERSION,
        "task": TASK,
        "primary_metric": PRIMARY_METRIC,
        "dataset_revision": DATASET_REVISION,
        "models": models,
        "expected_examples_per_model": args.expected_examples_per_model,
        "complete_run_gate": True,
        "model_results": model_rows,
        "macro_results": macro_rows,
        "depth_macro_results": depth_rows,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(args.out_dir / "summary.json")


if __name__ == "__main__":
    main()
