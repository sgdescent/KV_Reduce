#!/usr/bin/env python3
"""Rebuild a KV sensitivity profile from a prefix of per-example rows."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from acceptance_risk_statistics import paired_drop_statistics, zero_drop_statistics


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fields: List[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _as_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _mean(rows: Sequence[Dict[str, Any]], field: str) -> float | None:
    values = [_as_float(row.get(field)) for row in rows]
    finite = [value for value in values if value is not None]
    return statistics.mean(finite) if finite else None


def _sorted_ids(rows: Iterable[Dict[str, Any]], index_field: str) -> List[int]:
    return sorted({int(row[index_field]) for row in rows})


def _quality_profile(
    profile_rows: Sequence[Dict[str, Any]],
    raw_rows: Sequence[Dict[str, Any]],
    *,
    selected_ids: set[int],
    risk_metric: str,
) -> List[Dict[str, Any]]:
    by_candidate: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        if int(row["sequence_idx"]) in selected_ids:
            by_candidate[str(row["candidate"])].append(row)

    result = []
    for source in profile_rows:
        row = dict(source)
        candidate = str(row["candidate"])
        candidate_rows = by_candidate.get(candidate, [])
        if len(candidate_rows) != len(selected_ids):
            raise ValueError(
                f"Candidate {candidate!r} has {len(candidate_rows)} selected rows; "
                f"expected {len(selected_ids)}."
            )
        for field in (
            "native_nll",
            "quantized_nll",
            "delta_nll",
            "kl_p_to_q",
            "kl_q_to_p",
            "js",
            "tv",
            "accept_mass",
            "top1_match",
            "top5_overlap",
            "draft_prob_on_target_top1",
            "target_prob_on_target_top1",
            "affected_token_fraction",
        ):
            value = _mean(candidate_rows, field)
            if value is not None:
                row[field] = value
        if risk_metric == "kl":
            risk = _as_float(row.get("kl_p_to_q")) or 0.0
        elif risk_metric == "js":
            risk = _as_float(row.get("js")) or 0.0
        else:
            risk = _as_float(row.get("delta_nll")) or 0.0
        row["quality_risk"] = max(0.0, risk)
        delta_nll = _as_float(row.get("delta_nll")) or 0.0
        row["perplexity_ratio"] = math.exp(delta_nll)
        result.append(row)
    return result


def _invalid_acceptance_ids(
    rows: Sequence[Dict[str, Any]],
    *,
    selected_ids: set[int],
    tie_margin: float,
) -> set[int]:
    invalid = set()
    for row in rows:
        prompt_idx = int(row["prompt_idx"])
        if prompt_idx not in selected_ids:
            continue
        if _as_float(row.get("matches_target_greedy")) == 1.0:
            continue
        margin = _as_float(row.get("mismatch_min_top1_margin"))
        if margin is None or margin > tie_margin:
            invalid.add(prompt_idx)
    return invalid


def _acceptance_profile(
    profile_rows: Sequence[Dict[str, Any]],
    raw_rows: Sequence[Dict[str, Any]],
    *,
    valid_ids: set[int],
    baseline_config: str,
    z_score: float,
) -> List[Dict[str, Any]]:
    by_candidate: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        if int(row["prompt_idx"]) in valid_ids:
            by_candidate[str(row["config"])].append(row)
    if baseline_config not in by_candidate:
        raise ValueError(f"Baseline {baseline_config!r} is missing from selected prompt rows.")
    baseline_rows = by_candidate[baseline_config]
    baseline_accept_rate = _mean(baseline_rows, "accept_rate") or 0.0
    baseline_accept_mass = _mean(baseline_rows, "round_accept_mass") or 0.0

    result = []
    for source in profile_rows:
        row = dict(source)
        candidate = str(row["candidate"])
        candidate_rows = by_candidate.get(candidate, [])
        if len(candidate_rows) != len(valid_ids):
            raise ValueError(
                f"Candidate {candidate!r} has {len(candidate_rows)} valid rows; "
                f"expected {len(valid_ids)}."
            )
        accept_rate = _mean(candidate_rows, "accept_rate") or 0.0
        accept_mass = _mean(candidate_rows, "round_accept_mass") or 0.0
        row.update(
            {
                "accept_rate": accept_rate,
                "accept_rate_drop": baseline_accept_rate - accept_rate,
                "accept_mass": accept_mass,
                "accept_mass_drop": baseline_accept_mass - accept_mass,
                "accepted_per_round": _mean(candidate_rows, "accepted_per_round") or 0.0,
                "round_js": _mean(candidate_rows, "round_js") or 0.0,
                "round_top1_match": _mean(candidate_rows, "round_top1_match") or 0.0,
            }
        )
        if candidate == baseline_config:
            row.update(zero_drop_statistics())
            row.update(zero_drop_statistics("accept_mass_drop"))
        else:
            row.update(
                paired_drop_statistics(
                    baseline_rows,
                    candidate_rows,
                    metric="accept_rate",
                    prefix="accept_rate_drop",
                    z_score=z_score,
                )
            )
            row.update(
                paired_drop_statistics(
                    baseline_rows,
                    candidate_rows,
                    metric="round_accept_mass",
                    prefix="accept_mass_drop",
                    z_score=z_score,
                )
            )
        result.append(row)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile_csv", required=True, type=Path)
    parser.add_argument("--raw_csv", required=True, type=Path)
    parser.add_argument("--objective", required=True, choices=("quality", "acceptance"))
    parser.add_argument("--num_samples", required=True, type=int)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--quality_risk_metric", choices=("kl", "js", "delta_nll"), default="kl")
    parser.add_argument("--baseline_config", default="baseline_none")
    parser.add_argument("--exactness_tie_margin", type=float, default=1e-3)
    parser.add_argument("--z_score", type=float, default=1.96)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.num_samples <= 0:
        raise ValueError("num_samples must be positive.")
    profile_rows = read_csv(args.profile_csv)
    raw_rows = read_csv(args.raw_csv)
    index_field = "sequence_idx" if args.objective == "quality" else "prompt_idx"
    available_ids = _sorted_ids(raw_rows, index_field)
    if len(available_ids) < args.num_samples:
        raise ValueError(
            f"Requested {args.num_samples} samples but only {len(available_ids)} are available."
        )
    selected_ids = set(available_ids[: args.num_samples])
    invalid_ids: set[int] = set()
    if args.objective == "quality":
        rebuilt = _quality_profile(
            profile_rows,
            raw_rows,
            selected_ids=selected_ids,
            risk_metric=args.quality_risk_metric,
        )
        valid_ids = selected_ids
    else:
        invalid_ids = _invalid_acceptance_ids(
            raw_rows,
            selected_ids=selected_ids,
            tie_margin=args.exactness_tie_margin,
        )
        valid_ids = selected_ids - invalid_ids
        if not valid_ids:
            raise ValueError("Every selected acceptance prompt failed the exactness audit.")
        rebuilt = _acceptance_profile(
            profile_rows,
            raw_rows,
            valid_ids=valid_ids,
            baseline_config=args.baseline_config,
            z_score=args.z_score,
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "profile_summary.csv", rebuilt)
    payload = {
        "objective": args.objective,
        "source_profile_csv": str(args.profile_csv),
        "source_raw_csv": str(args.raw_csv),
        "requested_num_samples": args.num_samples,
        "available_num_samples": len(available_ids),
        "selected_ids": sorted(selected_ids),
        "valid_ids": sorted(valid_ids),
        "excluded_non_tie_ids": sorted(invalid_ids),
        "num_candidates": len(rebuilt),
        "quality_risk_metric": args.quality_risk_metric,
        "exactness_tie_margin": args.exactness_tie_margin,
        "z_score": args.z_score,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"Wrote {len(rebuilt)} candidates from {len(valid_ids)}/{args.num_samples} "
        f"valid {args.objective} samples to {args.out_dir}"
    )


if __name__ == "__main__":
    main()
