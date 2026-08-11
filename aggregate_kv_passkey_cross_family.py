#!/usr/bin/env python3
"""Strict cross-family meta-analysis for confusable passkey KV quantization."""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


EVALUATOR_VERSION = "kv_multiple_choice_cached_v2"
GENERATOR_VERSION = "synthetic_associative_passkey_v3"


def parse_sources(values: Sequence[str]) -> Dict[str, Path]:
    sources: Dict[str, Path] = {}
    for value in values:
        label, separator, raw_path = value.partition("=")
        if not separator or not label.strip() or not raw_path.strip():
            raise ValueError(f"Invalid source {value!r}; expected LABEL=SUMMARY.json")
        label = label.strip()
        if label in sources:
            raise ValueError(f"Duplicate source label: {label}")
        sources[label] = Path(raw_path.strip())
    return sources


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    values = list(rows)
    if not values:
        return
    fields: List[str] = []
    for row in values:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(values)


def bootstrap_model_ci(
    values: Sequence[float], *, seed: int, samples: int
) -> Dict[str, float]:
    if not values:
        raise ValueError("Cannot bootstrap an empty model set")
    if len(values) == 1:
        value = float(values[0])
        return {"mean": value, "ci_low": value, "ci_high": value}
    rng = random.Random(seed)
    estimates = sorted(
        statistics.mean(values[rng.randrange(len(values))] for _ in values)
        for _ in range(samples)
    )
    return {
        "mean": statistics.mean(values),
        "ci_low": estimates[int(0.025 * samples)],
        "ci_high": estimates[min(samples - 1, int(0.975 * samples))],
    }


def validate_summary(
    summary: Mapping[str, Any],
    *,
    path: Path,
    expected_contexts: Sequence[int],
    expected_runs: int,
) -> None:
    failures = []
    if summary.get("evaluator_version") != EVALUATOR_VERSION:
        failures.append("evaluator")
    if summary.get("task_generator_version") != GENERATOR_VERSION:
        failures.append("generator")
    if summary.get("expected_passkey_variant") != "confusable_records":
        failures.append("variant")
    if summary.get("expected_passkey_score") != "normalized":
        failures.append("score")
    if int(summary.get("expected_num_choices", -1)) != 16:
        failures.append("choices")
    if list(map(int, summary.get("expected_contexts", []))) != list(expected_contexts):
        failures.append("contexts")
    if int(summary.get("num_complete_runs", -1)) != expected_runs:
        failures.append("run_count")
    if summary.get("complete_run_gate") is not True:
        failures.append("complete")
    if summary.get("missing_runs"):
        failures.append("missing")
    if summary.get("underfilled_runs"):
        failures.append("underfilled")
    if failures:
        raise ValueError(f"Invalid passkey summary {path}: {','.join(failures)}")


def collect_rows(
    sources: Mapping[str, Path],
    *,
    expected_contexts: Sequence[int],
    expected_runs: int,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    accuracy_rows: List[Dict[str, Any]] = []
    contrast_rows: List[Dict[str, Any]] = []
    for model, path in sources.items():
        summary = json.loads(path.read_text(encoding="utf-8"))
        validate_summary(
            summary,
            path=path,
            expected_contexts=expected_contexts,
            expected_runs=expected_runs,
        )
        for row in summary["grouped"]:
            if row["depth"] != "all":
                continue
            accuracy_rows.append({"model": model, **row})
        for row in summary["comparisons"]:
            if (
                row["depth"] == "all"
                and row["config_a"] == "k4v2"
                and row["config_b"] == "k2v4"
            ):
                contrast_rows.append({"model": model, **row})
    expected_contrasts = {
        (model, context) for model in sources for context in expected_contexts
    }
    observed_contrasts = {
        (str(row["model"]), int(row["context"])) for row in contrast_rows
    }
    missing = sorted(expected_contrasts - observed_contrasts)
    if missing:
        raise ValueError(f"Missing K4V2-vs-K2V4 contrasts: {missing}")
    return accuracy_rows, contrast_rows


def aggregate_contrasts(
    rows: Iterable[Mapping[str, Any]], *, bootstrap_samples: int, seed: int
) -> List[Dict[str, Any]]:
    grouped: Dict[int, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["context"])].append(row)
    output = []
    for context, values in sorted(grouped.items()):
        effect = bootstrap_model_ci(
            [float(row["accuracy_a_minus_b_mean"]) for row in values],
            seed=seed + context,
            samples=bootstrap_samples,
        )
        output.append(
            {
                "context": context,
                "num_models": len(values),
                "models": ";".join(sorted(str(row["model"]) for row in values)),
                "k4v2_minus_k2v4_macro_mean": effect["mean"],
                "k4v2_minus_k2v4_model_bootstrap_ci_low": effect["ci_low"],
                "k4v2_minus_k2v4_model_bootstrap_ci_high": effect["ci_high"],
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--expected_contexts", default="8192,16384,32768")
    parser.add_argument("--expected_seeds", default="0,1,2")
    parser.add_argument("--bootstrap_samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    sources = parse_sources(args.source)
    expected_contexts = [int(value) for value in args.expected_contexts.split(",")]
    expected_seeds = [int(value) for value in args.expected_seeds.split(",")]
    expected_runs = len(expected_contexts) * len(expected_seeds)
    accuracy_rows, contrast_rows = collect_rows(
        sources,
        expected_contexts=expected_contexts,
        expected_runs=expected_runs,
    )
    macro = aggregate_contrasts(
        contrast_rows,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "model_accuracy.csv", accuracy_rows)
    write_csv(args.out_dir / "model_contrasts.csv", contrast_rows)
    write_csv(args.out_dir / "macro_contrasts.csv", macro)
    payload = {
        "evaluator_version": EVALUATOR_VERSION,
        "task_generator_version": GENERATOR_VERSION,
        "expected_contexts": expected_contexts,
        "expected_seeds": expected_seeds,
        "num_models": len(sources),
        "models": sorted(sources),
        "complete_gate": True,
        "macro_contrasts": macro,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Aggregated {len(sources)} complete passkey model families")
    print(args.out_dir / "summary.json")


if __name__ == "__main__":
    main()
