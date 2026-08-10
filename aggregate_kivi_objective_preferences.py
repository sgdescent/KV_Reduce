#!/usr/bin/env python3
"""Aggregate paired precision-allocation preferences under two objectives."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from spec_kv_statistics import bootstrap_acceptance_contrast
from aggregate_value_precision_sweep import parse_seed_filter


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fields: List[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_mean_ci(values: List[float], *, seed: int, samples: int = 10_000) -> Dict[str, float]:
    if not values:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    if len(values) == 1:
        return {"mean": values[0], "ci_low": values[0], "ci_high": values[0]}
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


def exact_enough(row: Dict[str, str], tie_margin: float) -> bool:
    if float(row.get("matches_target_greedy", 0.0)) >= 0.5:
        return True
    try:
        margin = float(row.get("mismatch_min_top1_margin", "nan"))
    except (TypeError, ValueError):
        return False
    return math.isfinite(margin) and margin <= tie_margin


def load_spec_rows(
    root: Path, seeds: set[int] | None = None
) -> Tuple[Dict[Tuple[int, int, str], Dict[str, Dict[str, str]]], Dict[Tuple[int, str], List[float]]]:
    grouped: Dict[Tuple[int, int, str], Dict[str, Dict[str, str]]] = {}
    memory: Dict[Tuple[int, str], List[float]] = defaultdict(list)
    for seed_dir in sorted(root.glob("ctx_*/seed_*")):
        seed_hint = int(seed_dir.name.removeprefix("seed_"))
        if seeds is not None and seed_hint not in seeds:
            continue
        summary_path = seed_dir / "summary.json"
        rows_path = seed_dir / "benchmark_rows.csv"
        if not summary_path.exists() or not rows_path.exists():
            continue
        summary = read_json(summary_path)
        if summary.get("runtime", {}).get("evaluator_version") != "cached_dynamic_v4":
            raise ValueError(f"Stale speculative evaluator in {summary_path}")
        context = int(summary["config"]["prompt_len"])
        seed = int(summary["config"]["seed"])
        if seeds is not None and seed not in seeds:
            continue
        for row in read_csv(rows_path):
            grouped.setdefault((context, seed, row["prompt_idx"]), {})[row["config"]] = row
        for config, metrics in summary["summaries"].items():
            memory[(context, config)].append(float(metrics["total_cache_saved_fraction"]))
    return grouped, memory


def load_quality_rows(
    root: Path, seeds: set[int] | None = None
) -> Dict[Tuple[int, int, str], Dict[str, Dict[str, str]]]:
    grouped: Dict[Tuple[int, int, str], Dict[str, Dict[str, str]]] = {}
    for seed_dir in sorted(root.glob("ctx_*/seed_*")):
        seed_hint = int(seed_dir.name.removeprefix("seed_"))
        if seeds is not None and seed_hint not in seeds:
            continue
        summary_path = seed_dir / "summary.json"
        rows_path = seed_dir / "raw_sequence_rows.csv"
        if not summary_path.exists() or not rows_path.exists():
            continue
        summary = read_json(summary_path)
        if summary.get("runtime", {}).get("evaluator_version") != "teacher_forced_cached_v1":
            raise ValueError(f"Stale quality evaluator in {summary_path}")
        context = int(summary["config"]["prompt_len"])
        seed = int(summary["config"]["seed"])
        if seeds is not None and seed not in seeds:
            continue
        for row in read_csv(rows_path):
            grouped.setdefault((context, seed, row["sequence_idx"]), {})[row["candidate"]] = row
    return grouped


def parse_pairs(values: str) -> List[Tuple[str, str]]:
    pairs = []
    for item in values.split(";"):
        if not item.strip():
            continue
        fields = [field.strip() for field in item.split(",")]
        if len(fields) != 2 or not all(fields):
            raise ValueError(f"Invalid config pair: {item!r}")
        pairs.append((fields[0], fields[1]))
    return pairs


def preference_label(spec_mean: float, quality_mean: float, config_a: str, config_b: str) -> Dict[str, Any]:
    spec_preference = config_a if spec_mean > 0 else config_b if spec_mean < 0 else "tie"
    quality_preference = config_a if quality_mean < 0 else config_b if quality_mean > 0 else "tie"
    return {
        "spec_preference": spec_preference,
        "quality_preference": quality_preference,
        "preference_reversal": spec_preference != "tie"
        and quality_preference != "tie"
        and spec_preference != quality_preference,
    }


def resolved_preference_label(
    *,
    spec_ci: Dict[str, float],
    quality_ci: Dict[str, float],
    config_a: str,
    config_b: str,
) -> Dict[str, Any]:
    if spec_ci["ci_low"] > 0:
        spec_preference = config_a
    elif spec_ci["ci_high"] < 0:
        spec_preference = config_b
    else:
        spec_preference = "unresolved"

    # Lower KL is better, so a negative A-minus-B interval favors A.
    if quality_ci["ci_high"] < 0:
        quality_preference = config_a
    elif quality_ci["ci_low"] > 0:
        quality_preference = config_b
    else:
        quality_preference = "unresolved"

    return {
        "spec_preference_resolved": spec_preference,
        "quality_preference_resolved": quality_preference,
        "resolved_preference_reversal": spec_preference != "unresolved"
        and quality_preference != "unresolved"
        and spec_preference != quality_preference,
    }


def aggregate_preferences(
    *,
    spec_rows: Dict[Tuple[int, int, str], Dict[str, Dict[str, str]]],
    quality_rows: Dict[Tuple[int, int, str], Dict[str, Dict[str, str]]],
    memory: Dict[Tuple[int, str], List[float]],
    pairs: Iterable[Tuple[str, str]],
    tie_margin: float,
    max_memory_gap: float = 0.002,
) -> List[Dict[str, Any]]:
    contexts = sorted({key[0] for key in spec_rows} & {key[0] for key in quality_rows})
    output = []
    for context in contexts:
        for config_a, config_b in pairs:
            spec_pairs = []
            for key, configs in spec_rows.items():
                if key[0] != context or config_a not in configs or config_b not in configs:
                    continue
                if not exact_enough(configs[config_a], tie_margin) or not exact_enough(configs[config_b], tie_margin):
                    continue
                spec_pairs.append((configs[config_a], configs[config_b]))
            quality_kl_differences = []
            quality_nll_differences = []
            for key, configs in quality_rows.items():
                if key[0] != context or config_a not in configs or config_b not in configs:
                    continue
                quality_kl_differences.append(
                    float(configs[config_a]["kl_p_to_q"]) - float(configs[config_b]["kl_p_to_q"])
                )
                quality_nll_differences.append(
                    float(configs[config_a]["delta_nll"]) - float(configs[config_b]["delta_nll"])
                )
            if not spec_pairs or not quality_kl_differences:
                continue
            seed = context + sum(map(ord, config_a + config_b))
            spec_ci = bootstrap_acceptance_contrast(spec_pairs, (1.0, -1.0), seed=seed)
            quality_kl_ci = bootstrap_mean_ci(quality_kl_differences, seed=seed + 1)
            quality_nll_ci = bootstrap_mean_ci(quality_nll_differences, seed=seed + 2)
            config_a_saved = statistics.mean(memory[(context, config_a)])
            config_b_saved = statistics.mean(memory[(context, config_b)])
            memory_gap = abs(config_a_saved - config_b_saved)
            output.append(
                {
                    "context": context,
                    "config_a": config_a,
                    "config_b": config_b,
                    "spec_paired_count": len(spec_pairs),
                    "spec_acceptance_a_minus_b_mean": spec_ci["mean"],
                    "spec_acceptance_a_minus_b_ci_low": spec_ci["ci_low"],
                    "spec_acceptance_a_minus_b_ci_high": spec_ci["ci_high"],
                    "quality_paired_count": len(quality_kl_differences),
                    "quality_kl_a_minus_b_mean": quality_kl_ci["mean"],
                    "quality_kl_a_minus_b_ci_low": quality_kl_ci["ci_low"],
                    "quality_kl_a_minus_b_ci_high": quality_kl_ci["ci_high"],
                    "quality_delta_nll_a_minus_b_mean": quality_nll_ci["mean"],
                    "quality_delta_nll_a_minus_b_ci_low": quality_nll_ci["ci_low"],
                    "quality_delta_nll_a_minus_b_ci_high": quality_nll_ci["ci_high"],
                    "config_a_total_saved_fraction": config_a_saved,
                    "config_b_total_saved_fraction": config_b_saved,
                    "absolute_total_saved_fraction_gap": memory_gap,
                    "memory_matched": memory_gap <= max_memory_gap,
                    **preference_label(spec_ci["mean"], quality_kl_ci["mean"], config_a, config_b),
                    **resolved_preference_label(
                        spec_ci=spec_ci,
                        quality_ci=quality_kl_ci,
                        config_a=config_a,
                        config_b=config_b,
                    ),
                }
            )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec_dir", required=True, type=Path)
    parser.add_argument("--quality_dir", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument(
        "--config_pairs",
        default="k8v4,k4v8;k8v3,k3v8;k4v3,k3v4;k4v2,k2v4;k3v2,k2v3",
    )
    parser.add_argument("--tie_margin", type=float, default=1e-3)
    parser.add_argument(
        "--seeds",
        default="",
        help="Optional comma-separated seed allowlist for provenance-safe partial aggregation.",
    )
    parser.add_argument(
        "--max_memory_gap",
        type=float,
        default=0.002,
        help="Maximum absolute total-cache savings gap for a pair to count as memory matched.",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    selected_seeds = parse_seed_filter(args.seeds)
    spec_rows, memory = load_spec_rows(args.spec_dir, selected_seeds)
    quality_rows = load_quality_rows(args.quality_dir, selected_seeds)
    rows = aggregate_preferences(
        spec_rows=spec_rows,
        quality_rows=quality_rows,
        memory=memory,
        pairs=parse_pairs(args.config_pairs),
        tie_margin=args.tie_margin,
        max_memory_gap=args.max_memory_gap,
    )
    if not rows:
        raise ValueError("No matched objective-preference rows were found.")
    write_csv(args.out_dir / "paired_preferences.csv", rows)
    payload = {
        "num_comparisons": len(rows),
        "num_preference_reversals": sum(bool(row["preference_reversal"]) for row in rows),
        "num_memory_matched_comparisons": sum(bool(row["memory_matched"]) for row in rows),
        "num_memory_matched_preference_reversals": sum(
            bool(row["memory_matched"] and row["preference_reversal"]) for row in rows
        ),
        "num_resolved_preference_reversals": sum(
            bool(row["resolved_preference_reversal"]) for row in rows
        ),
        "num_memory_matched_resolved_preference_reversals": sum(
            bool(row["memory_matched"] and row["resolved_preference_reversal"])
            for row in rows
        ),
        "max_memory_gap": args.max_memory_gap,
        "selected_seeds": sorted(selected_seeds) if selected_seeds is not None else None,
        "comparisons": rows,
    }
    (args.out_dir / "preference_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(args.out_dir / "preference_summary.json")


if __name__ == "__main__":
    main()
