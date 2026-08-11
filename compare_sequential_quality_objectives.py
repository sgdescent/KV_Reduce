#!/usr/bin/env python3
"""Compare exact speculative acceptance with matched ordinary-LM KV quality."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from compare_value_precision_objectives import (
    pareto_configs,
    select_max_savings,
    select_objective_choices,
    spearman,
)


SPEC_VERSION = "cached_dynamic_v6_sequential_target"
QUALITY_VERSION = "teacher_forced_cached_v1"


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def validate_provenance(spec: Dict[str, Any], quality: Dict[str, Any]) -> None:
    spec_runtime = spec.get("runtime", {})
    if spec_runtime.get("source_evaluator_version") != SPEC_VERSION:
        raise ValueError("Speculative aggregate is not from the exact sequential evaluator.")
    if spec_runtime.get("exactness_gate") != "all_rows_match_independent_target_greedy":
        raise ValueError("Speculative aggregate is missing the target-exactness gate.")
    if not spec_runtime.get("full_run_gate"):
        raise ValueError("Speculative aggregate is missing the full-run gate.")
    quality_runtime = quality.get("runtime", {})
    if quality_runtime.get("source_evaluator_version") != QUALITY_VERSION:
        raise ValueError("Quality aggregate has stale evaluator provenance.")
    if not quality_runtime.get("full_run_gate"):
        raise ValueError("Quality aggregate is missing the full-run gate.")


def matched_rows(
    spec: Dict[str, Any], quality: Dict[str, Any]
) -> List[Dict[str, Any]]:
    validate_provenance(spec, quality)
    spec_summaries = {
        (int(row["prompt_len"]), str(row["config"])): row
        for row in spec.get("macro_summaries", [])
        if row.get("config") != "none"
    }
    acceptance_deltas = {
        (int(row["prompt_len"]), str(row["left_config"])): row
        for row in spec.get("macro_contrasts", [])
        if row.get("right_config") == "none"
    }
    quality_summaries = {
        (int(row["context"]), str(row["config"])): row
        for row in quality.get("grouped", [])
    }
    keys = sorted(spec_summaries.keys() & acceptance_deltas.keys() & quality_summaries.keys())
    output: List[Dict[str, Any]] = []
    for key in keys:
        spec_row = spec_summaries[key]
        delta = acceptance_deltas[key]
        quality_row = quality_summaries[key]
        output.append(
            {
                "context": key[0],
                "config": key[1],
                "k_bits": quality_row["k_bits"],
                "v_bits": quality_row["v_bits"],
                "total_cache_saved_fraction": spec_row["total_cache_saved_fraction"],
                "draft_cache_saved_fraction": spec_row["draft_cache_saved_fraction"],
                "acceptance_rate": spec_row["acceptance_rate"],
                "acceptance_delta_mean": delta["acceptance_difference"],
                "acceptance_delta_ci_low": delta["ci_low"],
                "acceptance_delta_ci_high": delta["ci_high"],
                "quality_kl_mean": quality_row["kl_p_to_q_mean"],
                "quality_kl_ci_low": quality_row["kl_p_to_q_ci_low"],
                "quality_kl_ci_high": quality_row["kl_p_to_q_ci_high"],
                "quality_delta_nll_mean": quality_row["delta_nll_mean"],
                "quality_top1_match_mean": quality_row["top1_match_mean"],
            }
        )
    if not output:
        raise ValueError("Exact speculative and quality aggregates have no matched rows.")
    return output


def paired_objective_contrasts(
    spec: Dict[str, Any], quality: Dict[str, Any]
) -> List[Dict[str, Any]]:
    quality_pairs = {
        (int(row["context"]), str(row["config_a"]), str(row["config_b"])): row
        for row in quality.get("paired_precision_contrasts", [])
    }
    output: List[Dict[str, Any]] = []
    for row in spec.get("macro_contrasts", []):
        left = str(row["left_config"])
        right = str(row["right_config"])
        key = (int(row["prompt_len"]), left, right)
        quality_row = quality_pairs.get(key)
        if quality_row is None:
            continue
        spec_low = float(row["ci_low"])
        spec_high = float(row["ci_high"])
        quality_low = float(quality_row["kl_contrast_ci_low"])
        quality_high = float(quality_row["kl_contrast_ci_high"])
        spec_preference = left if spec_low > 0 else right if spec_high < 0 else "unresolved"
        quality_preference = left if quality_high < 0 else right if quality_low > 0 else "unresolved"
        output.append(
            {
                "context": key[0],
                "config_a": left,
                "config_b": right,
                "spec_acceptance_a_minus_b": row["acceptance_difference"],
                "spec_ci_low": spec_low,
                "spec_ci_high": spec_high,
                "quality_kl_a_minus_b": quality_row["kl_contrast_mean"],
                "quality_ci_low": quality_low,
                "quality_ci_high": quality_high,
                "spec_preference": spec_preference,
                "quality_preference": quality_preference,
                "resolved_preference_reversal": spec_preference not in {"unresolved", quality_preference}
                and quality_preference != "unresolved",
            }
        )
    return output


def summarize_contexts(
    rows: Iterable[Dict[str, Any]], *, savings_targets: Sequence[float]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["context"])].append(row)
    summaries: List[Dict[str, Any]] = []
    choices: List[Dict[str, Any]] = []
    pareto_rows: List[Dict[str, Any]] = []
    for context, context_rows in sorted(grouped.items()):
        acceptance_harm = [-float(row["acceptance_delta_mean"]) for row in context_rows]
        quality_harm = [float(row["quality_kl_mean"]) for row in context_rows]
        spec_frontier = pareto_configs(
            [{**row, "acceptance_harm": -float(row["acceptance_delta_mean"])} for row in context_rows],
            harm_key="acceptance_harm",
        )
        quality_frontier = pareto_configs(context_rows, harm_key="quality_kl_mean")
        for config in sorted(set(spec_frontier) | set(quality_frontier)):
            pareto_rows.append(
                {
                    "context": context,
                    "config": config,
                    "on_spec_frontier": config in spec_frontier,
                    "on_quality_frontier": config in quality_frontier,
                }
            )
        context_choices = []
        for target in savings_targets:
            choice = select_objective_choices(context_rows, minimum_savings=target)
            if choice is not None:
                choice = {"context": context, **choice}
                context_choices.append(choice)
                choices.append(choice)
        summaries.append(
            {
                "context": context,
                "num_configs": len(context_rows),
                "spearman_acceptance_harm_vs_quality_kl": spearman(
                    acceptance_harm, quality_harm
                ),
                "spec_acceptance_pareto_configs": spec_frontier,
                "quality_kl_pareto_configs": quality_frontier,
                "objective_choices": context_choices,
                **select_max_savings(
                    context_rows,
                    acceptance_drop_budget=0.02,
                    quality_kl_budget=0.01,
                ),
            }
        )
    return summaries, choices, pareto_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec_summary", type=Path, required=True)
    parser.add_argument("--quality_summary", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--savings_targets", default="0.20,0.25,0.30,0.33")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    spec = read_json(args.spec_summary)
    quality = read_json(args.quality_summary)
    rows = matched_rows(spec, quality)
    targets = [float(value) for value in args.savings_targets.split(",") if value.strip()]
    contexts, choices, pareto = summarize_contexts(rows, savings_targets=targets)
    contrasts = paired_objective_contrasts(spec, quality)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "matched_objectives.csv", rows)
    write_csv(args.out_dir / "paired_objective_contrasts.csv", contrasts)
    write_csv(args.out_dir / "objective_choices.csv", choices)
    write_csv(args.out_dir / "objective_pareto.csv", pareto)
    payload = {
        "runtime": {
            "spec_evaluator_version": SPEC_VERSION,
            "quality_evaluator_version": QUALITY_VERSION,
            "exactness_gate": "all_rows_match_independent_target_greedy",
            "full_run_gate": True,
        },
        "num_matched_rows": len(rows),
        "num_resolved_preference_reversals": sum(
            bool(row["resolved_preference_reversal"]) for row in contrasts
        ),
        "contexts": contexts,
        "paired_objective_contrasts": contrasts,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(args.out_dir / "summary.json")


if __name__ == "__main__":
    main()
