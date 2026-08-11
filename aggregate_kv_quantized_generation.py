#!/usr/bin/env python3
"""Aggregate paired free-running KV-quantized generation shards."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

try:
    import numpy as np
except ImportError:  # Keep the report aggregator usable in minimal environments.
    np = None


EXPECTED_VERSION = "free_running_cached_v1"
METRICS = (
    "exact_sequence_match",
    "token_match_fraction",
    "prefix_retained_fraction",
    "first_divergence_is_bf16_tie",
)
CONTRASTS = (
    ("k8v4", "k4v8"),
    ("k4v3", "k3v4"),
    ("k4v2", "k2v4"),
    ("k4v4", "none"),
)


def percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("Cannot compute a percentile of an empty sequence.")
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    rng: random.Random,
    samples: int,
) -> Tuple[float, float, float]:
    if not values:
        raise ValueError("Cannot bootstrap an empty sequence.")
    mean = sum(values) / len(values)
    if len(values) == 1 or samples <= 0:
        return mean, mean, mean
    if np is None:
        draws = [
            sum(values[rng.randrange(len(values))] for _ in values) / len(values)
            for _ in range(samples)
        ]
        return mean, percentile(draws, 0.025), percentile(draws, 0.975)
    array = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(rng.getrandbits(64))
    draws = np.empty(samples, dtype=np.float64)
    chunk_size = min(samples, 4096)
    for start in range(0, samples, chunk_size):
        stop = min(samples, start + chunk_size)
        indices = generator.integers(0, len(array), size=(stop - start, len(array)))
        draws[start:stop] = array[indices].mean(axis=1)
    low, high = np.quantile(draws, (0.025, 0.975))
    return mean, float(low), float(high)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_runs(pattern: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    runs: List[Dict[str, Any]] = []
    summary_names = sorted(
        {
            name
            for item in pattern.split(";")
            if item.strip()
            for name in glob.glob(item.strip(), recursive=True)
        }
    )
    for summary_name in summary_names:
        summary_path = Path(summary_name)
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        version = payload.get("runtime", {}).get("evaluator_version")
        if version != EXPECTED_VERSION:
            raise ValueError(f"{summary_path} has evaluator_version={version!r}.")
        row_path = summary_path.with_name("rows.csv")
        if not row_path.exists():
            raise FileNotFoundError(row_path)
        run_id = str(summary_path.parent)
        with row_path.open("r", encoding="utf-8", newline="") as handle:
            run_rows = list(csv.DictReader(handle))
        observed = int(payload.get("observed_prompts", 0))
        if observed <= 0 or not run_rows:
            raise ValueError(f"{summary_path} contains no observed prompts.")
        runs.append(
            {
                "run_id": run_id,
                "summary_path": str(summary_path),
                "observed_prompts": observed,
                "model": payload.get("config", {}).get("model"),
                "prompt_len": payload.get("config", {}).get("prompt_len"),
                "max_new_tokens": payload.get("config", {}).get("max_new_tokens"),
                "seed": payload.get("config", {}).get("seed"),
                "skip_prompts": payload.get("config", {}).get("skip_prompts"),
            }
        )
        for row in run_rows:
            row["run_id"] = run_id
            row["prompt_key"] = f"{run_id}:{row['prompt_idx']}"
            rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No summaries matched {pattern!r}.")
    return rows, runs


def aggregate(
    rows: Iterable[Dict[str, Any]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    grouped: Dict[Tuple[str, int, int, str], List[Dict[str, Any]]] = defaultdict(list)
    paired: Dict[Tuple[str, int, int], Dict[str, Dict[str, Dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for row in rows:
        key = (
            str(row["model"]),
            int(row["prompt_len"]),
            int(row["max_new_tokens"]),
            str(row["config"]),
        )
        grouped[key].append(row)
        base = key[:3]
        paired[base][str(row["prompt_key"])][str(row["config"])] = row

    rng = random.Random(seed)
    summaries: List[Dict[str, Any]] = []
    for (model, prompt_len, max_new_tokens, config), items in sorted(grouped.items()):
        output: Dict[str, Any] = {
            "model": model,
            "prompt_len": prompt_len,
            "max_new_tokens": max_new_tokens,
            "config": config,
            "num_prompt_occurrences": len(items),
            "num_runs": len({row["run_id"] for row in items}),
            "cache_saved_fraction": sum(float(row["cache_saved_fraction"]) for row in items)
            / len(items),
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in items]
            mean, low, high = bootstrap_mean_ci(
                values,
                rng=rng,
                samples=bootstrap_samples,
            )
            output[metric] = mean
            output[f"{metric}_ci_low"] = low
            output[f"{metric}_ci_high"] = high
        summaries.append(output)

    contrasts: List[Dict[str, Any]] = []
    for (model, prompt_len, max_new_tokens), prompt_rows in sorted(paired.items()):
        for left, right in CONTRASTS:
            common = [
                configs
                for configs in prompt_rows.values()
                if left in configs and right in configs
            ]
            if not common:
                continue
            for metric in METRICS[:3]:
                differences = [
                    float(configs[left][metric]) - float(configs[right][metric])
                    for configs in common
                ]
                mean, low, high = bootstrap_mean_ci(
                    differences,
                    rng=rng,
                    samples=bootstrap_samples,
                )
                contrasts.append(
                    {
                        "model": model,
                        "prompt_len": prompt_len,
                        "max_new_tokens": max_new_tokens,
                        "left_config": left,
                        "right_config": right,
                        "metric": metric,
                        "num_paired_prompts": len(differences),
                        "difference": mean,
                        "ci_low": low,
                        "ci_high": high,
                    }
                )
    return summaries, contrasts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary_glob", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--bootstrap_samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=8675309)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows, runs = load_runs(args.summary_glob)
    summaries, contrasts = aggregate(
        rows,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "generation_summary.csv", summaries)
    write_csv(out_dir / "paired_contrasts.csv", contrasts)
    payload = {
        "runtime": {"source_evaluator_version": EXPECTED_VERSION},
        "runs": runs,
        "summaries": summaries,
        "paired_contrasts": contrasts,
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Aggregated {len(runs)} runs and {len(rows)} rows into {out_dir}.")


if __name__ == "__main__":
    main()
