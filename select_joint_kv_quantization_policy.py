#!/usr/bin/env python3
"""Select role-aware target/draft KV precision under quality constraints."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def quality_index(summary: Mapping[str, Any]) -> Dict[Tuple[int, str], Dict[str, Any]]:
    index: Dict[Tuple[int, str], Dict[str, Any]] = {}
    contexts = {int(row["context"]) for row in summary.get("grouped", [])}
    for context in contexts:
        index[(context, "none")] = {
            "kl_p_to_q_mean": 0.0,
            "kl_p_to_q_ci_high": 0.0,
            "delta_nll_mean": 0.0,
            "delta_nll_ci_high": 0.0,
            "top1_match_mean": 1.0,
            "accept_mass_mean": 1.0,
        }
    for row in summary.get("grouped", []):
        index[(int(row["context"]), str(row["config"]))] = dict(row)
    return index


def evaluate_candidates(
    joint_summary: Mapping[str, Any],
    target_quality_summary: Mapping[str, Any],
    *,
    target_kl_max: float,
    target_delta_nll_max: float,
    target_top1_min: float,
    acceptance_drop_max: float,
) -> List[Dict[str, Any]]:
    quality = quality_index(target_quality_summary)
    evaluated: List[Dict[str, Any]] = []
    for raw in joint_summary.get("grouped", []):
        row = dict(raw)
        context = int(row["context"])
        target_config = str(row["target_config"])
        target_quality = quality.get((context, target_config))
        reasons: List[str] = []
        if target_quality is None:
            reasons.append("missing_target_quality")
            target_quality = {}

        kl_high = target_quality.get("kl_p_to_q_ci_high")
        nll_high = target_quality.get("delta_nll_ci_high")
        top1 = target_quality.get("top1_match_mean")
        acceptance_low = row.get("paired_acceptance_delta_ci_low")
        if not finite(kl_high) or float(kl_high) > target_kl_max:
            reasons.append("target_kl")
        if not finite(nll_high) or float(nll_high) > target_delta_nll_max:
            reasons.append("target_delta_nll")
        if not finite(top1) or float(top1) < target_top1_min:
            reasons.append("target_top1")
        if not finite(acceptance_low) or float(acceptance_low) < -acceptance_drop_max:
            reasons.append("acceptance")

        row.update(
            {
                "target_quality_kl_mean": target_quality.get("kl_p_to_q_mean"),
                "target_quality_kl_ci_high": kl_high,
                "target_quality_delta_nll_mean": target_quality.get(
                    "delta_nll_mean"
                ),
                "target_quality_delta_nll_ci_high": nll_high,
                "target_quality_top1_match_mean": top1,
                "target_quality_accept_mass_mean": target_quality.get(
                    "accept_mass_mean"
                ),
                "feasible": not reasons,
                "constraint_failures": ";".join(reasons),
            }
        )
        evaluated.append(row)
    return evaluated


def select_by_context(rows: Iterable[Mapping[str, Any]]) -> Dict[int, Dict[str, Any]]:
    by_context: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("feasible"):
            by_context[int(row["context"])].append(dict(row))

    selected: Dict[int, Dict[str, Any]] = {}
    for context, candidates in by_context.items():
        candidates.sort(
            key=lambda row: (
                float(row["total_cache_saved_fraction"]),
                float(row["paired_acceptance_delta_ci_low"]),
                -float(row["target_quality_kl_ci_high"]),
            ),
            reverse=True,
        )
        selected[context] = candidates[0]
    return selected


def make_plot(rows: List[Dict[str, Any]], out_dir: Path) -> List[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    contexts = sorted({int(row["context"]) for row in rows})
    fig, axes = plt.subplots(
        1, len(contexts), figsize=(5.6 * len(contexts), 4.8), squeeze=False
    )
    for axis, context in zip(axes[0], contexts):
        subset = [row for row in rows if int(row["context"]) == context]
        for feasible, color, label in (
            (False, "#B8B8B8", "Infeasible"),
            (True, "#188977", "Feasible"),
        ):
            values = [row for row in subset if bool(row["feasible"]) is feasible]
            if values:
                axis.scatter(
                    [100.0 * float(row["total_cache_saved_fraction"]) for row in values],
                    [
                        100.0 * float(row["paired_acceptance_delta_mean"])
                        for row in values
                    ],
                    color=color,
                    label=label,
                    s=55,
                    edgecolors="#222222",
                    linewidths=0.4,
                )
        axis.axhline(0.0, color="#222222", linewidth=1)
        axis.set_title(f"Context {context:,}")
        axis.set_xlabel("Total target + draft KV saved (%)")
        axis.set_ylabel("Acceptance change vs BF16 (pp)")
        axis.grid(alpha=0.22)
        axis.legend()
    fig.suptitle("Quality-Constrained Joint KV Precision", fontweight="bold")
    fig.tight_layout()
    paths = []
    for extension in ("png", "pdf"):
        path = out_dir / f"joint_policy_selection.{extension}"
        fig.savefig(path, dpi=240, bbox_inches="tight")
        paths.append(str(path))
    plt.close(fig)
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint_summary", required=True, type=Path)
    parser.add_argument("--target_quality_summary", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--target_kl_max", type=float, default=0.01)
    parser.add_argument("--target_delta_nll_max", type=float, default=0.02)
    parser.add_argument("--target_top1_min", type=float, default=0.95)
    parser.add_argument("--acceptance_drop_max", type=float, default=0.02)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    joint = read_json(args.joint_summary)
    quality = read_json(args.target_quality_summary)
    rows = evaluate_candidates(
        joint,
        quality,
        target_kl_max=args.target_kl_max,
        target_delta_nll_max=args.target_delta_nll_max,
        target_top1_min=args.target_top1_min,
        acceptance_drop_max=args.acceptance_drop_max,
    )
    if not rows:
        raise ValueError("No joint target/draft candidates were found.")
    selected = select_by_context(rows)
    write_csv(args.out_dir / "candidate_constraints.csv", rows)
    payload = {
        "joint_summary": str(args.joint_summary),
        "target_quality_summary": str(args.target_quality_summary),
        "constraints": {
            "target_kl_ci_high_max": args.target_kl_max,
            "target_delta_nll_ci_high_max": args.target_delta_nll_max,
            "target_top1_match_min": args.target_top1_min,
            "acceptance_delta_ci_low_min": -args.acceptance_drop_max,
        },
        "num_candidates": len(rows),
        "num_feasible": sum(bool(row["feasible"]) for row in rows),
        "selected_by_context": {str(key): value for key, value in selected.items()},
        "plots": make_plot(rows, args.out_dir),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Evaluated {len(rows)} candidates; {payload['num_feasible']} feasible")
    print(args.out_dir / "summary.json")


if __name__ == "__main__":
    main()
