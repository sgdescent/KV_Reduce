#!/usr/bin/env python3
"""Aggregate matched KIVI objective results across model families."""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


DEFAULT_PAIRS = (
    "qwen25_3b_15b",
    "qwen25_7b_3b",
    "qwen3_8b_4b",
    "llama31_8b_llama32_3b",
    "olmo2_7b_1b",
    "smollm2_17b_360m",
)


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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


def bootstrap_macro_ci(
    values: Sequence[float], *, seed: int, samples: int = 10_000
) -> Dict[str, float]:
    if not values:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    if len(values) == 1:
        value = float(values[0])
        return {"mean": value, "ci_low": value, "ci_high": value}
    rng = random.Random(seed)
    estimates = sorted(
        statistics.mean(float(values[rng.randrange(len(values))]) for _ in values)
        for _ in range(samples)
    )
    return {
        "mean": statistics.mean(map(float, values)),
        "ci_low": estimates[int(0.025 * samples)],
        "ci_high": estimates[min(samples - 1, int(0.975 * samples))],
    }


def collect_pair_rows(
    root: Path,
    expected_pairs: Sequence[str],
    *,
    aggregate_name: str = "aggregate",
    comparison_name: str = "",
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    matched: List[Dict[str, Any]] = []
    preferences: List[Dict[str, Any]] = []
    exactness = Counter()
    missing = []
    run_counts: Dict[str, Dict[str, int]] = {}
    for pair in expected_pairs:
        comparison_dir = root / "comparison" / pair
        if comparison_name:
            comparison_dir = comparison_dir / comparison_name
        spec_summary_path = root / "spec" / pair / aggregate_name / "summary.json"
        quality_summary_path = root / "quality" / pair / aggregate_name / "summary.json"
        matched_path = comparison_dir / "matched_objectives.csv"
        preference_path = comparison_dir / "paired_preferences.csv"
        required = (spec_summary_path, quality_summary_path, matched_path, preference_path)
        absent = [str(path) for path in required if not path.exists()]
        if absent:
            missing.extend(absent)
            continue

        spec_summary = read_json(spec_summary_path)
        quality_summary = read_json(quality_summary_path)
        run_counts[pair] = {
            "spec": int(spec_summary.get("num_complete_runs", 0)),
            "quality": int(quality_summary.get("num_complete_runs", 0)),
        }
        exactness.update(spec_summary.get("exactness", {}))
        exactness["invalid_prompt_occurrences"] += int(
            spec_summary.get("invalid_prompt_occurrences", 0)
        )
        for row in read_csv(matched_path):
            matched.append({"pair": pair, **row})
        for row in read_csv(preference_path):
            preferences.append({"pair": pair, **row})
    audit = {
        "expected_pairs": list(expected_pairs),
        "complete_pairs": sorted({str(row["pair"]) for row in matched}),
        "missing_artifacts": missing,
        "run_counts": run_counts,
        "exactness": dict(exactness),
    }
    return matched, preferences, audit


def aggregate_configs(
    rows: Iterable[Mapping[str, Any]],
    *,
    acceptance_drop_budget: float,
    quality_kl_budget: float,
) -> List[Dict[str, Any]]:
    grouped: Dict[tuple[int, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["context"]), str(row["config"]))].append(row)

    output: List[Dict[str, Any]] = []
    for (context, config), values in sorted(grouped.items()):
        acceptance = bootstrap_macro_ci(
            [float(row["acceptance_delta_mean"]) for row in values],
            seed=context + sum(map(ord, config)),
        )
        quality = bootstrap_macro_ci(
            [float(row["quality_kl_mean"]) for row in values],
            seed=2 * context + sum(map(ord, config)),
        )
        output.append(
            {
                "context": context,
                "config": config,
                "num_pairs": len(values),
                "pairs": ";".join(sorted(str(row["pair"]) for row in values)),
                "total_cache_saved_fraction_macro_mean": statistics.mean(
                    float(row["total_cache_saved_fraction"]) for row in values
                ),
                "draft_cache_saved_fraction_macro_mean": statistics.mean(
                    float(row["draft_cache_saved_fraction"]) for row in values
                ),
                "acceptance_delta_macro_mean": acceptance["mean"],
                "acceptance_delta_macro_ci_low": acceptance["ci_low"],
                "acceptance_delta_macro_ci_high": acceptance["ci_high"],
                "quality_kl_macro_mean": quality["mean"],
                "quality_kl_macro_ci_low": quality["ci_low"],
                "quality_kl_macro_ci_high": quality["ci_high"],
                "quality_top1_match_macro_mean": statistics.mean(
                    float(row["quality_top1_match_mean"]) for row in values
                ),
                "num_pairs_acceptance_conservative": sum(
                    float(row["acceptance_delta_ci_low"]) >= -acceptance_drop_budget
                    for row in values
                ),
                "num_pairs_quality_conservative": sum(
                    float(row["quality_kl_ci_high"]) <= quality_kl_budget
                    for row in values
                ),
                "num_pairs_jointly_conservative": sum(
                    float(row["acceptance_delta_ci_low"]) >= -acceptance_drop_budget
                    and float(row["quality_kl_ci_high"]) <= quality_kl_budget
                    for row in values
                ),
            }
        )
    return output


def summarize_preferences(rows: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    values = list(rows)
    matched = [
        row
        for row in values
        if str(row.get("memory_matched", "")).lower() in {"true", "1"}
    ]
    reversals = [
        row
        for row in values
        if str(row.get("preference_reversal", "")).lower() in {"true", "1"}
    ]
    matched_reversals = [row for row in reversals if row in matched]
    return {
        "num_comparisons": len(values),
        "num_preference_reversals": len(reversals),
        "num_memory_matched_comparisons": len(matched),
        "num_memory_matched_preference_reversals": len(matched_reversals),
        "reversal_rows": reversals,
        "memory_matched_reversal_rows": matched_reversals,
    }


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
        for row in subset:
            axis.errorbar(
                100.0 * float(row["total_cache_saved_fraction_macro_mean"]),
                100.0 * float(row["acceptance_delta_macro_mean"]),
                yerr=[
                    [
                        100.0
                        * (
                            float(row["acceptance_delta_macro_mean"])
                            - float(row["acceptance_delta_macro_ci_low"])
                        )
                    ],
                    [
                        100.0
                        * (
                            float(row["acceptance_delta_macro_ci_high"])
                            - float(row["acceptance_delta_macro_mean"])
                        )
                    ],
                ],
                marker="o",
                capsize=3,
                color="#26456E",
            )
            axis.annotate(
                str(row["config"]),
                (
                    100.0 * float(row["total_cache_saved_fraction_macro_mean"]),
                    100.0 * float(row["acceptance_delta_macro_mean"]),
                ),
                fontsize=8,
            )
        axis.axhline(0.0, color="#222222", linewidth=1)
        axis.set_title(f"Context {context:,}")
        axis.set_xlabel("Macro-average total KV saved (%)")
        axis.set_ylabel("Macro acceptance change (pp)")
        axis.grid(alpha=0.22)
    fig.suptitle("Cross-Family K/V Precision", fontweight="bold")
    fig.tight_layout()
    paths = []
    for extension in ("png", "pdf"):
        path = out_dir / f"cross_family_kivi_meta.{extension}"
        fig.savefig(path, dpi=240, bbox_inches="tight")
        paths.append(str(path))
    plt.close(fig)
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("outputs/kivi_objective_cross_family")
    )
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--expected_pairs", default=",".join(DEFAULT_PAIRS))
    parser.add_argument(
        "--aggregate_name",
        default="aggregate",
        help="Per-pair aggregate subdirectory, useful for explicitly labeled partial analyses.",
    )
    parser.add_argument(
        "--comparison_name",
        default="",
        help="Optional subdirectory below comparison/<pair>.",
    )
    parser.add_argument("--acceptance_drop_budget", type=float, default=0.02)
    parser.add_argument("--quality_kl_budget", type=float, default=0.01)
    parser.add_argument("--allow_incomplete", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    expected = [value.strip() for value in args.expected_pairs.split(",") if value.strip()]
    matched, preferences, audit = collect_pair_rows(
        args.root,
        expected,
        aggregate_name=args.aggregate_name,
        comparison_name=args.comparison_name,
    )
    if audit["missing_artifacts"] and not args.allow_incomplete:
        raise ValueError(
            "Missing cross-family artifacts:\n" + "\n".join(audit["missing_artifacts"])
        )
    if not matched:
        raise ValueError("No complete cross-family matched-objective rows were found.")
    grouped = aggregate_configs(
        matched,
        acceptance_drop_budget=args.acceptance_drop_budget,
        quality_kl_budget=args.quality_kl_budget,
    )
    preference_summary = summarize_preferences(preferences)
    write_csv(args.out_dir / "pair_results.csv", matched)
    write_csv(args.out_dir / "cross_family_configs.csv", grouped)
    write_csv(args.out_dir / "preference_rows.csv", preferences)
    payload = {
        "acceptance_drop_budget": args.acceptance_drop_budget,
        "quality_kl_budget": args.quality_kl_budget,
        "audit": audit,
        "preference_summary": preference_summary,
        "grouped": grouped,
        "plots": make_plot(grouped, args.out_dir),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Aggregated {len(audit['complete_pairs'])}/{len(expected)} model pairs; "
        f"{preference_summary['num_memory_matched_preference_reversals']} "
        "memory-matched reversals"
    )
    print(args.out_dir / "summary.json")


if __name__ == "__main__":
    main()
