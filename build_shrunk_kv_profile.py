#!/usr/bin/env python3
"""Shrink noisy acceptance sensitivities toward an ordinary-quality prior."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Sequence, Tuple


PROFILE_VERSION = "quality_prior_shrinkage_v1"


def read_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def profile_key(row: Dict[str, str]) -> Tuple[int, str, int]:
    return int(float(row["layer"])), str(row["component"]), int(float(row["bits"]))


def eligible(rows: Iterable[Dict[str, str]]) -> Iterable[Dict[str, str]]:
    for row in rows:
        if row.get("component") not in {"k", "v"}:
            continue
        if int(float(row.get("layer", -1))) < 0:
            continue
        yield row


def nonnegative_float(row: Dict[str, str], field: str) -> float:
    value = float(row[field])
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {field} for {profile_key(row)}")
    return max(0.0, value)


def calibrate_prior_scale(quality: Sequence[float], observed: Sequence[float]) -> float:
    denominator = sum(value * value for value in quality)
    if denominator <= 0.0:
        return 0.0
    return max(0.0, sum(q * a for q, a in zip(quality, observed)) / denominator)


def estimate_residual_variance(
    observed: Sequence[float],
    prior: Sequence[float],
    standard_errors: Sequence[float],
    *,
    mode: str = "debiased",
) -> float:
    if not observed:
        return 0.0
    residuals = [value - prediction for value, prediction in zip(observed, prior)]
    centered = [value - mean(residuals) for value in residuals]
    empirical = sum(value * value for value in centered) / max(1, len(centered) - 1)
    if mode == "empirical":
        return empirical
    if mode != "debiased":
        raise ValueError("variance mode must be 'empirical' or 'debiased'.")
    noise = mean(error * error for error in standard_errors)
    return max(0.0, empirical - noise)


def shrink_rows(
    quality_rows: Sequence[Dict[str, str]],
    acceptance_rows: Sequence[Dict[str, str]],
    *,
    quality_field: str,
    acceptance_field: str,
    acceptance_se_field: str,
    prior_strength: float,
    ucb_z: float,
    variance_mode: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    if prior_strength < 0.0:
        raise ValueError("prior_strength must be nonnegative.")
    quality_by_key = {profile_key(row): row for row in eligible(quality_rows)}
    acceptance_by_key = {profile_key(row): row for row in eligible(acceptance_rows)}
    common = sorted(set(quality_by_key) & set(acceptance_by_key))
    if not common:
        raise ValueError("Quality and acceptance profiles contain no common candidates.")

    quality = [nonnegative_float(quality_by_key[key], quality_field) for key in common]
    observed = [nonnegative_float(acceptance_by_key[key], acceptance_field) for key in common]
    standard_errors = [
        nonnegative_float(acceptance_by_key[key], acceptance_se_field) for key in common
    ]
    prior_scale = calibrate_prior_scale(quality, observed)
    prior = [prior_scale * value for value in quality]
    residual_variance = estimate_residual_variance(
        observed,
        prior,
        standard_errors,
        mode=variance_mode,
    )

    output: List[Dict[str, Any]] = []
    for key, quality_risk, observed_risk, standard_error, prior_risk in zip(
        common, quality, observed, standard_errors, prior
    ):
        noise_variance = prior_strength * standard_error * standard_error
        denominator = residual_variance + noise_variance
        reliability = residual_variance / denominator if denominator > 0.0 else 0.0
        posterior = reliability * observed_risk + (1.0 - reliability) * prior_risk
        posterior_variance = (
            residual_variance * noise_variance / denominator if denominator > 0.0 else 0.0
        )
        posterior_se = math.sqrt(max(0.0, posterior_variance))
        row: Dict[str, Any] = dict(acceptance_by_key[key])
        row.update(
            {
                "quality_prior_risk": prior_risk,
                "quality_prior_scale": prior_scale,
                "observed_acceptance_risk": observed_risk,
                "observed_acceptance_se": standard_error,
                "acceptance_reliability": reliability,
                "shrunk_acceptance_risk": max(0.0, posterior),
                "shrunk_acceptance_posterior_se": posterior_se,
                "shrunk_acceptance_ucb": max(0.0, posterior + ucb_z * posterior_se),
            }
        )
        output.append(row)

    diagnostics = {
        "num_common_candidates": float(len(common)),
        "quality_prior_scale": prior_scale,
        "residual_variance": residual_variance,
        "mean_observed_acceptance_risk": mean(observed),
        "mean_quality_prior_risk": mean(prior),
        "mean_reliability": mean(float(row["acceptance_reliability"]) for row in output),
    }
    return output, diagnostics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality_profile_csv", required=True)
    parser.add_argument("--acceptance_profile_csv", required=True)
    parser.add_argument("--quality_field", default="quality_risk")
    parser.add_argument("--acceptance_field", default="accept_rate_drop_prompt_mean")
    parser.add_argument("--acceptance_se_field", default="accept_rate_drop_prompt_se")
    parser.add_argument("--prior_strength", type=float, default=1.0)
    parser.add_argument("--ucb_z", type=float, default=0.0)
    parser.add_argument(
        "--variance_mode",
        choices=["empirical", "debiased"],
        default="empirical",
        help="Empirical mode yields a tunable shrinkage path; debiased mode may collapse to the prior.",
    )
    parser.add_argument("--out_dir", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output, diagnostics = shrink_rows(
        read_rows(args.quality_profile_csv),
        read_rows(args.acceptance_profile_csv),
        quality_field=args.quality_field,
        acceptance_field=args.acceptance_field,
        acceptance_se_field=args.acceptance_se_field,
        prior_strength=args.prior_strength,
        ucb_z=args.ucb_z,
        variance_mode=args.variance_mode,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in output:
        for key in row:
            if key not in fields:
                fields.append(key)
    with (out_dir / "profile_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output)
    payload = {
        "runtime": {
            "profile_version": PROFILE_VERSION,
            "quality_field": args.quality_field,
            "acceptance_field": args.acceptance_field,
            "acceptance_se_field": args.acceptance_se_field,
            "prior_strength": args.prior_strength,
            "ucb_z": args.ucb_z,
            "variance_mode": args.variance_mode,
        },
        "diagnostics": diagnostics,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
