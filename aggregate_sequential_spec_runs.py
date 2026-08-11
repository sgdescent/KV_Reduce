#!/usr/bin/env python3
"""Aggregate exact sequential-target speculative-decoding runs."""

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
except ImportError:  # Small local tests can use the dependency-free fallback.
    np = None


EXPECTED_VERSION = "cached_dynamic_v6_sequential_target"
CONTRASTS = (("k8v4", "k4v8"), ("k4v4", "none"), ("k4v8", "none"), ("k8v4", "none"))


def parse_patterns(value: str) -> List[str]:
    return [item.strip() for item in value.split(";") if item.strip()]


def ratio(numerators: Sequence[float], denominators: Sequence[float]) -> float:
    denominator = sum(denominators)
    return sum(numerators) / denominator if denominator > 0 else 0.0


def percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def bootstrap_ratio_ci(
    numerators: Sequence[float],
    denominators: Sequence[float],
    *,
    rng: random.Random,
    samples: int,
) -> Tuple[float, float, float]:
    if not numerators or len(numerators) != len(denominators):
        raise ValueError("Bootstrap inputs must be non-empty and aligned.")
    estimate = ratio(numerators, denominators)
    if len(numerators) == 1 or samples <= 0:
        return estimate, estimate, estimate
    if np is not None:
        numerator_array = np.asarray(numerators, dtype=np.float64)
        denominator_array = np.asarray(denominators, dtype=np.float64)
        generator = np.random.default_rng(rng.getrandbits(64))
        draws = np.empty(samples, dtype=np.float64)
        chunk_size = min(samples, 4096)
        for start in range(0, samples, chunk_size):
            stop = min(samples, start + chunk_size)
            indices = generator.integers(
                0, len(numerator_array), size=(stop - start, len(numerator_array))
            )
            sampled_denominators = denominator_array[indices].sum(axis=1)
            draws[start:stop] = np.divide(
                numerator_array[indices].sum(axis=1),
                sampled_denominators,
                out=np.zeros(stop - start, dtype=np.float64),
                where=sampled_denominators > 0,
            )
        low, high = np.quantile(draws, (0.025, 0.975))
        return estimate, float(low), float(high)

    draws: List[float] = []
    for _ in range(samples):
        indices = [rng.randrange(len(numerators)) for _ in numerators]
        draws.append(
            ratio(
                [numerators[index] for index in indices],
                [denominators[index] for index in indices],
            )
        )
    return estimate, percentile(draws, 0.025), percentile(draws, 0.975)


def bootstrap_paired_ratio_difference(
    left: Sequence[Tuple[float, float]],
    right: Sequence[Tuple[float, float]],
    *,
    rng: random.Random,
    samples: int,
) -> Tuple[float, float, float]:
    if not left or len(left) != len(right):
        raise ValueError("Paired bootstrap inputs must be non-empty and aligned.")
    left_n, left_d = zip(*left)
    right_n, right_d = zip(*right)
    estimate = ratio(left_n, left_d) - ratio(right_n, right_d)
    if len(left) == 1 or samples <= 0:
        return estimate, estimate, estimate
    draws: List[float] = []
    if np is not None:
        arrays = [np.asarray(values, dtype=np.float64) for values in (left_n, left_d, right_n, right_d)]
        generator = np.random.default_rng(rng.getrandbits(64))
        draw_array = np.empty(samples, dtype=np.float64)
        chunk_size = min(samples, 4096)
        for start in range(0, samples, chunk_size):
            stop = min(samples, start + chunk_size)
            indices = generator.integers(0, len(left), size=(stop - start, len(left)))
            ln, ld, rn, rd = (array[indices].sum(axis=1) for array in arrays)
            draw_array[start:stop] = ln / ld - rn / rd
        low, high = np.quantile(draw_array, (0.025, 0.975))
        return estimate, float(low), float(high)

    for _ in range(samples):
        indices = [rng.randrange(len(left)) for _ in left]
        draws.append(
            ratio([left_n[index] for index in indices], [left_d[index] for index in indices])
            - ratio([right_n[index] for index in indices], [right_d[index] for index in indices])
        )
    return estimate, percentile(draws, 0.025), percentile(draws, 0.975)


def load_runs(patterns: Iterable[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    paths = sorted({path for pattern in patterns for path in glob.glob(pattern, recursive=True)})
    if not paths:
        raise FileNotFoundError("No strict sequential-target summaries matched.")
    runs: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []
    for name in paths:
        path = Path(name)
        payload = json.loads(path.read_text(encoding="utf-8"))
        runtime = payload.get("runtime", {})
        if runtime.get("evaluator_version") != EXPECTED_VERSION:
            raise ValueError(f"{path} has evaluator_version={runtime.get('evaluator_version')!r}.")
        if runtime.get("target_verification_mode") != "sequential":
            raise ValueError(f"{path} does not use sequential target verification.")
        if payload.get("target_quant_configs") != ["none"]:
            raise ValueError(f"{path} must use an unquantized target cache.")
        row_path = path.with_name("benchmark_rows.csv")
        with row_path.open("r", encoding="utf-8", newline="") as handle:
            run_rows = list(csv.DictReader(handle))
        mismatches = [
            row
            for row in run_rows
            if float(row["matches_target_greedy"]) < 1.0 or int(row["first_target_mismatch"]) != -1
        ]
        if mismatches:
            raise ValueError(f"{path} contains {len(mismatches)} target-exactness failures.")
        config = payload.get("config", {})
        run_id = str(path.parent)
        runs.append(
            {
                "run_id": run_id,
                "summary_path": str(path),
                "big_model": config.get("big_model"),
                "small_model": config.get("small_model"),
                "prompt_len": int(config.get("prompt_len", 0)),
                "max_new_tokens": int(config.get("max_new_tokens", 0)),
                "seed": int(config.get("seed", 0)),
                "num_prompts": int(payload.get("num_prompts", 0)),
            }
        )
        memory = payload.get("memory_estimates", {})
        for row in run_rows:
            row["run_id"] = run_id
            row["prompt_key"] = f"{run_id}:{row['prompt_idx']}"
            row["big_model"] = config.get("big_model")
            row["small_model"] = config.get("small_model")
            row["prompt_len"] = int(config.get("prompt_len", 0))
            row["max_new_tokens"] = int(config.get("max_new_tokens", 0))
            row["draft_cache_saved_fraction"] = float(
                memory.get(row["config"], {}).get("draft_cache_saved_fraction", 0.0)
            )
            row["total_cache_saved_fraction"] = float(
                memory.get(row["config"], {}).get("total_cache_saved_fraction", 0.0)
            )
            rows.append(row)
    return rows, runs


def aggregate(
    rows: Sequence[Dict[str, Any]], *, bootstrap_samples: int, seed: int
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    grouped: Dict[Tuple[str, str, int, int, str], List[Dict[str, Any]]] = defaultdict(list)
    paired: Dict[Tuple[str, str, int, int], Dict[str, Dict[str, Dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for row in rows:
        key = (
            str(row["big_model"]),
            str(row["small_model"]),
            int(row["prompt_len"]),
            int(row["max_new_tokens"]),
            str(row["config"]),
        )
        grouped[key].append(row)
        paired[key[:4]][str(row["prompt_key"])][str(row["config"])] = row

    rng = random.Random(seed)
    summaries: List[Dict[str, Any]] = []
    for (big_model, small_model, prompt_len, max_new_tokens, config), items in sorted(grouped.items()):
        accepted = [float(row["accepted_tokens"]) for row in items]
        proposed = [float(row["proposed_tokens"]) for row in items]
        mean, low, high = bootstrap_ratio_ci(accepted, proposed, rng=rng, samples=bootstrap_samples)
        summaries.append(
            {
                "big_model": big_model,
                "small_model": small_model,
                "prompt_len": prompt_len,
                "max_new_tokens": max_new_tokens,
                "config": config,
                "num_paired_prompts": len(items),
                "acceptance_rate": mean,
                "acceptance_ci_low": low,
                "acceptance_ci_high": high,
                "draft_cache_saved_fraction": sum(float(row["draft_cache_saved_fraction"]) for row in items) / len(items),
                "total_cache_saved_fraction": sum(float(row["total_cache_saved_fraction"]) for row in items) / len(items),
                "exact_target_match_fraction": 1.0,
            }
        )

    contrasts: List[Dict[str, Any]] = []
    for (big_model, small_model, prompt_len, max_new_tokens), prompt_rows in sorted(paired.items()):
        for left_name, right_name in CONTRASTS:
            common = [configs for configs in prompt_rows.values() if left_name in configs and right_name in configs]
            if not common:
                continue
            left = [(float(item[left_name]["accepted_tokens"]), float(item[left_name]["proposed_tokens"])) for item in common]
            right = [(float(item[right_name]["accepted_tokens"]), float(item[right_name]["proposed_tokens"])) for item in common]
            mean, low, high = bootstrap_paired_ratio_difference(
                left, right, rng=rng, samples=bootstrap_samples
            )
            contrasts.append(
                {
                    "big_model": big_model,
                    "small_model": small_model,
                    "prompt_len": prompt_len,
                    "max_new_tokens": max_new_tokens,
                    "left_config": left_name,
                    "right_config": right_name,
                    "num_paired_prompts": len(common),
                    "acceptance_difference": mean,
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    return summaries, contrasts


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary_glob", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--bootstrap_samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=8675309)
    args = parser.parse_args()

    rows, runs = load_runs(parse_patterns(args.summary_glob))
    summaries, contrasts = aggregate(rows, bootstrap_samples=args.bootstrap_samples, seed=args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "summary.csv", summaries)
    write_csv(out_dir / "paired_contrasts.csv", contrasts)
    payload = {
        "runtime": {
            "source_evaluator_version": EXPECTED_VERSION,
            "exactness_gate": "all_rows_match_independent_target_greedy",
        },
        "runs": runs,
        "summaries": summaries,
        "paired_contrasts": contrasts,
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Aggregated {len(runs)} exact runs and {len(rows)} rows into {out_dir}.")


if __name__ == "__main__":
    main()
