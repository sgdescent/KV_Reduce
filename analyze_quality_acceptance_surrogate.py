#!/usr/bin/env python3
"""Test whether ordinary-LM quality predicts speculative acceptance harm."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

from spec_kv_statistics import bootstrap_mean_ci


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
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


def average_ranks(values: Sequence[float]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def correlation(left: Sequence[float], right: Sequence[float]) -> float:
    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    if len(left_array) < 2 or left_array.std() == 0.0 or right_array.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(left_array, right_array)[0, 1])


def leave_one_pair_out_predictions(
    rows: Sequence[Mapping[str, Any]],
    *,
    ridge: float = 0.1,
) -> List[Dict[str, Any]]:
    """Fit only ordinary-quality features and hold out each model pair."""

    targets = np.asarray([float(row["acceptance_delta_mean"]) for row in rows])
    features = np.asarray(
        [
            [
                math.log10(float(row["quality_kl_mean"]) + 1e-8),
                float(row["quality_top1_match_mean"]),
            ]
            for row in rows
        ],
        dtype=float,
    )
    predictions = np.zeros(len(rows), dtype=float)
    pairs = [str(row["pair"]) for row in rows]
    for held_pair in sorted(set(pairs)):
        test = np.asarray([pair == held_pair for pair in pairs])
        train = ~test
        mean = features[train].mean(axis=0)
        std = features[train].std(axis=0) + 1e-8
        design = np.column_stack(
            [np.ones(len(rows)), (features - mean) / std]
        )
        penalty = np.diag([0.0, ridge, ridge])
        weights = np.linalg.solve(
            design[train].T @ design[train] + penalty,
            design[train].T @ targets[train],
        )
        predictions[test] = design[test] @ weights
    return [
        {
            "pair": str(row["pair"]),
            "context": int(row["context"]),
            "config": str(row["config"]),
            "acceptance_delta_actual": float(target),
            "acceptance_delta_predicted": float(prediction),
            "prediction_error": float(prediction - target),
        }
        for row, target, prediction in zip(rows, targets, predictions)
    ]


def prediction_metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    actual = np.asarray([float(row["acceptance_delta_actual"]) for row in rows])
    predicted = np.asarray([float(row["acceptance_delta_predicted"]) for row in rows])
    residual_sum = float(np.square(actual - predicted).sum())
    total_sum = float(np.square(actual - actual.mean()).sum())
    return {
        "rmse": float(np.sqrt(np.mean(np.square(actual - predicted)))),
        "mae": float(np.mean(np.abs(actual - predicted))),
        "r_squared": 1.0 - residual_sum / total_sum if total_sum else float("nan"),
        "pearson": correlation(actual, predicted),
    }


def quality_gate_metrics(
    rows: Iterable[Mapping[str, Any]],
    *,
    quality_kl_budget: float,
    acceptance_drop_budget: float,
) -> Dict[str, Any]:
    values = list(rows)
    selected = [float(row["quality_kl_mean"]) <= quality_kl_budget for row in values]
    safe = [
        float(row["acceptance_delta_mean"]) >= -acceptance_drop_budget
        for row in values
    ]
    true_positive = sum(is_selected and is_safe for is_selected, is_safe in zip(selected, safe))
    selected_count = sum(selected)
    safe_count = sum(safe)
    return {
        "quality_kl_budget": quality_kl_budget,
        "acceptance_drop_budget": acceptance_drop_budget,
        "num_cells": len(values),
        "num_selected": selected_count,
        "num_acceptance_safe": safe_count,
        "num_selected_and_safe": true_positive,
        "precision": true_positive / selected_count if selected_count else float("nan"),
        "recall": true_positive / safe_count if safe_count else float("nan"),
        "false_safe_count": sum(
            is_selected and not is_safe for is_selected, is_safe in zip(selected, safe)
        ),
    }


def make_plot(rows: Sequence[Mapping[str, Any]], out_dir: Path) -> List[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    pairs = sorted({str(row["pair"]) for row in rows})
    palette = plt.get_cmap("tab10")
    fig, axis = plt.subplots(figsize=(7.3, 4.9))
    for pair_idx, pair in enumerate(pairs):
        subset = [row for row in rows if row["pair"] == pair]
        axis.scatter(
            [float(row["quality_kl_mean"]) for row in subset],
            [-100.0 * float(row["acceptance_delta_mean"]) for row in subset],
            label=pair,
            color=palette(pair_idx),
            alpha=0.82,
            edgecolors="#222222",
            linewidths=0.35,
        )
    axis.axvline(0.01, color="#333333", linestyle="--", linewidth=1, label="KL budget")
    axis.axhline(2.0, color="#777777", linestyle=":", linewidth=1, label="2 pp budget")
    axis.set_xscale("log")
    axis.set_xlabel("Ordinary-LM KL from BF16 (lower is better)")
    axis.set_ylabel("Speculative acceptance loss (pp)")
    axis.set_title("Ordinary Quality Predicts Acceptance Risk", fontweight="bold")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=7, ncol=2, frameon=False)
    fig.tight_layout()
    paths = []
    for extension in ("png", "pdf"):
        path = out_dir / f"quality_acceptance_surrogate.{extension}"
        fig.savefig(path, dpi=240, bbox_inches="tight")
        paths.append(str(path))
    plt.close(fig)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair_results", type=Path, required=True)
    parser.add_argument("--meta_summary", type=Path, default=None)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--bootstrap_samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_csv(args.pair_results)
    if not rows:
        raise ValueError("No paired objective rows were found.")
    required = {
        "pair",
        "context",
        "config",
        "acceptance_delta_mean",
        "quality_kl_mean",
        "quality_top1_match_mean",
    }
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Pair-results table is missing columns: {sorted(missing)}")
    meta_summary_path = args.meta_summary or args.pair_results.with_name("summary.json")
    meta_summary = json.loads(meta_summary_path.read_text(encoding="utf-8"))
    audit = meta_summary.get("audit", {})
    if audit.get("missing_artifacts"):
        raise ValueError("The source objective aggregate is incomplete.")
    row_pairs = {str(row["pair"]) for row in rows}
    complete_pairs = set(map(str, audit.get("complete_pairs", [])))
    if row_pairs != complete_pairs:
        raise ValueError(
            "Pair-results and aggregate audit disagree: "
            f"rows={sorted(row_pairs)}, audit={sorted(complete_pairs)}"
        )

    acceptance_harm = [-float(row["acceptance_delta_mean"]) for row in rows]
    quality_kl = [float(row["quality_kl_mean"]) for row in rows]
    pair_spearman = []
    for pair in sorted({str(row["pair"]) for row in rows}):
        subset = [row for row in rows if row["pair"] == pair]
        pair_spearman.append(
            {
                "pair": pair,
                "spearman": correlation(
                    average_ranks([-float(row["acceptance_delta_mean"]) for row in subset]),
                    average_ranks([float(row["quality_kl_mean"]) for row in subset]),
                ),
            }
        )
    pair_spearman_macro = bootstrap_mean_ci(
        [float(row["spearman"]) for row in pair_spearman],
        seed=args.seed,
        samples=args.bootstrap_samples,
    )

    predictions = leave_one_pair_out_predictions(rows)
    gates = [
        quality_gate_metrics(
            rows,
            quality_kl_budget=kl_budget,
            acceptance_drop_budget=acceptance_budget,
        )
        for kl_budget in (0.0025, 0.005, 0.01, 0.02, 0.05)
        for acceptance_budget in (0.01, 0.02, 0.05)
    ]
    write_csv(args.out_dir / "pair_spearman.csv", pair_spearman)
    write_csv(args.out_dir / "leave_one_pair_out_predictions.csv", predictions)
    write_csv(args.out_dir / "quality_gate_sweep.csv", gates)
    payload = {
        "source": str(args.pair_results),
        "meta_summary": str(meta_summary_path),
        "source_audit": audit,
        "num_cells": len(rows),
        "num_model_pairs": len({str(row["pair"]) for row in rows}),
        "overall_pearson_harm_vs_log_kl": correlation(
            acceptance_harm,
            [math.log10(value + 1e-8) for value in quality_kl],
        ),
        "overall_spearman_harm_vs_kl": correlation(
            average_ranks(acceptance_harm), average_ranks(quality_kl)
        ),
        "pair_spearman": pair_spearman,
        "pair_spearman_macro": pair_spearman_macro,
        "leave_one_pair_out": prediction_metrics(predictions),
        "quality_gate_sweep": gates,
        "plots": make_plot(rows, args.out_dir),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(args.out_dir / "summary.json")


if __name__ == "__main__":
    main()
