#!/usr/bin/env python3
"""Aggregate held-out quality/acceptance results for shrinkage allocations."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


EVALUATOR_VERSION = "shrunk_allocator_cross_eval_v1"


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def bootstrap_ci(values: Sequence[float], *, seed: int, samples: int) -> Tuple[float, float, float]:
    if not values:
        return math.nan, math.nan, math.nan
    estimate = sum(values) / len(values)
    if len(values) == 1:
        return estimate, estimate, estimate
    rng = random.Random(seed)
    draws = [
        sum(values[rng.randrange(len(values))] for _ in values) / len(values)
        for _ in range(samples)
    ]
    return estimate, percentile(draws, 0.025), percentile(draws, 0.975)


def classify_exactness(row: Dict[str, str], tie_margin: float) -> str:
    if float(row.get("matches_target_greedy", 0.0)) >= 0.5:
        return "exact"
    try:
        margin = float(row.get("mismatch_min_top1_margin", "nan"))
    except ValueError:
        margin = math.nan
    if math.isfinite(margin) and margin <= tie_margin:
        return "numerical_tie"
    return "non_tie"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--quality_dir", required=True)
    parser.add_argument("--acceptance_dir", required=True)
    parser.add_argument("--tie_margin", type=float, default=1e-3)
    parser.add_argument("--bootstrap_samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--out_dir", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    bundle = json.loads(Path(args.bundle).read_text(encoding="utf-8"))
    quality_dir = Path(args.quality_dir)
    acceptance_dir = Path(args.acceptance_dir)
    quality = json.loads((quality_dir / "summary.json").read_text(encoding="utf-8"))
    acceptance = json.loads((acceptance_dir / "summary.json").read_text(encoding="utf-8"))
    if quality.get("runtime", {}).get("evaluator_version") != "teacher_forced_cached_v1":
        raise ValueError("Unexpected ordinary-quality evaluator provenance.")
    if acceptance.get("runtime", {}).get("evaluator_version") != "cached_dynamic_v4":
        raise ValueError("Unexpected speculative evaluator provenance.")

    policies = bundle["policies"]
    policy_names = [str(policy["name"]) for policy in policies]
    rows = []
    for policy in policies:
        name = str(policy["name"])
        quality_row = quality["summaries"][name]
        acceptance_row = acceptance["summaries"][name]
        rows.append(
            {
                **policy,
                "quality_delta_nll": quality_row["delta_nll"],
                "quality_kl": quality_row["kl_p_to_q"],
                "quality_top1_match": quality_row["top1_match"],
                "spec_accept_rate": acceptance_row["overall_accept_rate"],
                "spec_accepted_per_round": acceptance_row["accepted_per_round"],
                "spec_round_js": acceptance_row["round_js"],
                "draft_cache_saved_fraction": acceptance_row["draft_cache_saved_fraction"],
                "total_cache_saved_fraction": acceptance_row["total_cache_saved_fraction"],
            }
        )

    benchmark_rows = read_csv(acceptance_dir / "benchmark_rows.csv")
    by_prompt: Dict[str, Dict[str, Dict[str, str]]] = defaultdict(dict)
    exactness_counts = {"exact": 0, "numerical_tie": 0, "non_tie": 0}
    invalid_prompts = set()
    for row in benchmark_rows:
        if row["config"] not in policy_names:
            continue
        status = classify_exactness(row, args.tie_margin)
        exactness_counts[status] += 1
        if status == "non_tie":
            invalid_prompts.add(row["prompt_idx"])
        by_prompt[row["prompt_idx"]][row["config"]] = row

    reference_name = next(policy["name"] for policy in policies if policy["kind"] == "quality")
    native_name = next(policy["name"] for policy in policies if policy["kind"] == "native")
    effects = []
    for policy in policies:
        name = str(policy["name"])
        for reference in (native_name, reference_name):
            if name == reference:
                continue
            differences = [
                float(configs[name]["accept_rate"]) - float(configs[reference]["accept_rate"])
                for prompt_idx, configs in by_prompt.items()
                if prompt_idx not in invalid_prompts and name in configs and reference in configs
            ]
            estimate, low, high = bootstrap_ci(
                differences,
                seed=args.seed + len(effects),
                samples=args.bootstrap_samples,
            )
            effects.append(
                {
                    "policy": name,
                    "reference": reference,
                    "num_paired_prompts": len(differences),
                    "accept_rate_difference": estimate,
                    "ci_low": low,
                    "ci_high": high,
                }
            )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "runtime": {
            "evaluator_version": EVALUATOR_VERSION,
            "quality_source_version": "teacher_forced_cached_v1",
            "acceptance_source_version": "cached_dynamic_v4",
            "tie_margin": args.tie_margin,
        },
        "rows": rows,
        "paired_acceptance_effects": effects,
        "exactness": {
            "row_counts": exactness_counts,
            "invalid_prompt_count": len(invalid_prompts),
            "invalid_prompt_indices": sorted(int(value) for value in invalid_prompts),
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with (out_dir / "policies.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
