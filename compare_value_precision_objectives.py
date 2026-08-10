#!/usr/bin/env python3
"""Join matched speculative-acceptance and ordinary-LM value sweeps."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fields: List[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def rankdata(values: Iterable[float]) -> List[float]:
    """Return average ranks, including ties, without requiring scipy."""
    items = list(enumerate(float(value) for value in values))
    ordered = sorted(items, key=lambda item: item[1])
    ranks = [0.0] * len(ordered)
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and ordered[end][1] == ordered[cursor][1]:
            end += 1
        average_rank = 0.5 * ((cursor + 1) + end)
        for offset in range(cursor, end):
            ranks[ordered[offset][0]] = average_rank
        cursor = end
    return ranks


def pearson(x_values: Iterable[float], y_values: Iterable[float]) -> float:
    x = list(map(float, x_values))
    y = list(map(float, y_values))
    if len(x) != len(y) or len(x) < 2:
        return float("nan")
    x_mean = statistics.mean(x)
    y_mean = statistics.mean(y)
    numerator = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y))
    x_norm = math.sqrt(sum((a - x_mean) ** 2 for a in x))
    y_norm = math.sqrt(sum((b - y_mean) ** 2 for b in y))
    if x_norm == 0.0 or y_norm == 0.0:
        return float("nan")
    return numerator / (x_norm * y_norm)


def spearman(x_values: Iterable[float], y_values: Iterable[float]) -> float:
    return pearson(rankdata(x_values), rankdata(y_values))


def select_max_savings(
    rows: Iterable[Dict[str, Any]],
    *,
    acceptance_drop_budget: float,
    quality_kl_budget: float,
) -> Dict[str, Any]:
    values = list(rows)
    mean_feasible = [
        row
        for row in values
        if float(row["acceptance_delta_mean"]) >= -acceptance_drop_budget
        and float(row["quality_kl_mean"]) <= quality_kl_budget
    ]
    conservative_feasible = [
        row
        for row in values
        if float(row["acceptance_delta_ci_low"]) >= -acceptance_drop_budget
        and float(row["quality_kl_ci_high"]) <= quality_kl_budget
    ]

    def best(candidates: List[Dict[str, Any]]) -> Dict[str, Any] | None:
        if not candidates:
            return None
        row = max(candidates, key=lambda item: float(item["total_cache_saved_fraction"]))
        return {
            "config": row["config"],
            "total_cache_saved_fraction": row["total_cache_saved_fraction"],
            "acceptance_delta_mean": row["acceptance_delta_mean"],
            "acceptance_delta_ci_low": row["acceptance_delta_ci_low"],
            "acceptance_delta_ci_high": row["acceptance_delta_ci_high"],
            "quality_kl_mean": row["quality_kl_mean"],
            "quality_kl_ci_high": row["quality_kl_ci_high"],
        }

    return {
        "mean_feasible_count": len(mean_feasible),
        "conservative_feasible_count": len(conservative_feasible),
        "best_mean_feasible": best(mean_feasible),
        "best_conservative_feasible": best(conservative_feasible),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec_summary", required=True, type=Path)
    parser.add_argument("--quality_summary", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--acceptance_drop_budget", type=float, default=0.02)
    parser.add_argument("--quality_kl_budget", type=float, default=0.01)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    spec = read_json(args.spec_summary)
    quality = read_json(args.quality_summary)
    spec_rows = {
        (int(row["context"]), str(row["config"])): row for row in spec["grouped"]
    }
    quality_rows = {
        (int(row["context"]), str(row["config"])): row for row in quality["grouped"]
    }
    joined: List[Dict[str, Any]] = []
    for key in sorted(spec_rows.keys() & quality_rows.keys()):
        spec_row = spec_rows[key]
        quality_row = quality_rows[key]
        joined.append(
            {
                "context": key[0],
                "config": key[1],
                "k_bits": spec_row["k_bits"],
                "v_bits": spec_row["v_bits"],
                "total_cache_saved_fraction": spec_row["total_cache_saved_fraction"],
                "draft_cache_saved_fraction": spec_row["draft_cache_saved_fraction"],
                "acceptance_delta_mean": spec_row["paired_acceptance_delta_mean"],
                "acceptance_delta_ci_low": spec_row["paired_acceptance_delta_ci_low"],
                "acceptance_delta_ci_high": spec_row["paired_acceptance_delta_ci_high"],
                "quality_kl_mean": quality_row["kl_p_to_q_mean"],
                "quality_kl_ci_low": quality_row["kl_p_to_q_ci_low"],
                "quality_kl_ci_high": quality_row["kl_p_to_q_ci_high"],
                "quality_delta_nll_mean": quality_row["delta_nll_mean"],
                "quality_top1_match_mean": quality_row["top1_match_mean"],
            }
        )
    if not joined:
        raise ValueError("The speculative and quality summaries have no matched configurations.")

    by_context: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in joined:
        by_context[int(row["context"])].append(row)
    context_summaries = []
    for context, rows in sorted(by_context.items()):
        acceptance_harm = [-float(row["acceptance_delta_mean"]) for row in rows]
        quality_harm = [float(row["quality_kl_mean"]) for row in rows]
        context_summaries.append(
            {
                "context": context,
                "num_configs": len(rows),
                "spearman_acceptance_harm_vs_quality_kl": spearman(acceptance_harm, quality_harm),
                **select_max_savings(
                    rows,
                    acceptance_drop_budget=args.acceptance_drop_budget,
                    quality_kl_budget=args.quality_kl_budget,
                ),
            }
        )

    write_csv(args.out_dir / "matched_objectives.csv", joined)
    payload = {
        "acceptance_drop_budget": args.acceptance_drop_budget,
        "quality_kl_budget": args.quality_kl_budget,
        "num_matched_rows": len(joined),
        "contexts": context_summaries,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(args.out_dir / "summary.json")


if __name__ == "__main__":
    main()
