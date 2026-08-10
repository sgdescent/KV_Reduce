#!/usr/bin/env python3
"""Compute paired uncertainty-aware risks for speculative acceptance profiles."""

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List, Sequence


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def passes_target_exactness(row: Dict[str, Any], *, tie_margin: float = 1e-3) -> bool:
    """Keep exact rows and BF16 ties; reject resolved target-reference mismatches."""
    if "matches_target_greedy" not in row:
        return True
    if _finite_float(row.get("matches_target_greedy")) == 1.0:
        return True
    for field in (
        "mismatch_min_top1_margin",
        "mismatch_target_top1_margin",
        "mismatch_verifier_top1_margin",
    ):
        margin = _finite_float(row.get(field))
        if margin is not None:
            return margin <= tie_margin
    return False


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def paired_drop_statistics(
    baseline_rows: Sequence[Dict[str, Any]],
    candidate_rows: Sequence[Dict[str, Any]],
    *,
    metric: str = "accept_rate",
    prefix: str = "accept_rate_drop",
    z_score: float = 1.96,
    exactness_tie_margin: float | None = 1e-3,
) -> Dict[str, float]:
    baseline_source = {str(row["prompt_idx"]): row for row in baseline_rows}
    candidate_source = {str(row["prompt_idx"]): row for row in candidate_rows}
    shared_before_audit = sorted(set(baseline_source) & set(candidate_source), key=int)
    if exactness_tie_margin is None:
        shared = shared_before_audit
    else:
        shared = [
            prompt_idx
            for prompt_idx in shared_before_audit
            if passes_target_exactness(
                baseline_source[prompt_idx],
                tie_margin=exactness_tie_margin,
            )
            and passes_target_exactness(
                candidate_source[prompt_idx],
                tie_margin=exactness_tie_margin,
            )
        ]
    if not shared:
        raise ValueError("No paired prompt rows were found for acceptance-risk estimation.")
    drops = [
        float(baseline_source[prompt_idx][metric]) - float(candidate_source[prompt_idx][metric])
        for prompt_idx in shared
    ]
    estimate = mean(drops)
    standard_error = stdev(drops) / math.sqrt(len(drops)) if len(drops) > 1 else 0.0
    lower = estimate - z_score * standard_error
    upper = estimate + z_score * standard_error
    return {
        f"{prefix}_prompt_mean": estimate,
        f"{prefix}_prompt_se": standard_error,
        f"{prefix}_lcb95": lower,
        f"{prefix}_ucb95": upper,
        f"{prefix}_ucb95_clipped": max(0.0, upper),
        "paired_prompt_count": float(len(drops)),
        "paired_prompt_count_before_exactness_audit": float(len(shared_before_audit)),
        "excluded_non_tie_prompt_count": float(len(shared_before_audit) - len(shared)),
    }


def zero_drop_statistics(prefix: str = "accept_rate_drop") -> Dict[str, float]:
    return {
        f"{prefix}_prompt_mean": 0.0,
        f"{prefix}_prompt_se": 0.0,
        f"{prefix}_lcb95": 0.0,
        f"{prefix}_ucb95": 0.0,
        f"{prefix}_ucb95_clipped": 0.0,
        "paired_prompt_count": 0.0,
        "paired_prompt_count_before_exactness_audit": 0.0,
        "excluded_non_tie_prompt_count": 0.0,
    }


def augment_profile(
    profile_rows: Sequence[Dict[str, Any]],
    raw_rows: Sequence[Dict[str, Any]],
    *,
    baseline_config: str,
    z_score: float,
) -> List[Dict[str, Any]]:
    by_config: Dict[str, List[Dict[str, Any]]] = {}
    for row in raw_rows:
        by_config.setdefault(str(row["config"]), []).append(row)
    if baseline_config not in by_config:
        raise ValueError(f"Baseline config {baseline_config!r} is missing from raw prompt rows.")

    augmented = []
    metric_specs = (
        ("accept_rate", "accept_rate_drop"),
        ("round_accept_mass", "accept_mass_drop"),
    )
    for source_row in profile_rows:
        row = dict(source_row)
        candidate = str(row["candidate"])
        if candidate == baseline_config:
            for _, prefix in metric_specs:
                row.update(zero_drop_statistics(prefix))
        else:
            if candidate not in by_config:
                raise ValueError(f"Candidate {candidate!r} is missing from raw prompt rows.")
            for metric, prefix in metric_specs:
                row.update(paired_drop_statistics(
                    by_config[baseline_config],
                    by_config[candidate],
                    metric=metric,
                    prefix=prefix,
                    z_score=z_score,
                ))
        augmented.append(row)
    return augmented


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Add paired uncertainty-aware risk fields to a SpecDec profile.")
    parser.add_argument("--profile_csv", required=True)
    parser.add_argument("--raw_prompt_csv", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--out_json", default=None)
    parser.add_argument("--baseline_config", default="baseline_none")
    parser.add_argument("--z_score", type=float, default=1.96)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    augmented = augment_profile(
        read_csv(Path(args.profile_csv)),
        read_csv(Path(args.raw_prompt_csv)),
        baseline_config=args.baseline_config,
        z_score=args.z_score,
    )
    out_csv = Path(args.out_csv)
    write_csv(augmented, out_csv)
    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(
            json.dumps(
                {
                    "profile_csv": args.profile_csv,
                    "raw_prompt_csv": args.raw_prompt_csv,
                    "baseline_config": args.baseline_config,
                    "z_score": args.z_score,
                    "num_rows": len(augmented),
                    "risk_fields": [
                        "accept_rate_drop_ucb95_clipped",
                        "accept_mass_drop_ucb95_clipped",
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    print(f"Wrote {len(augmented)} augmented rows to {out_csv}")


if __name__ == "__main__":
    main()
