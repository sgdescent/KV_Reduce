#!/usr/bin/env python3
"""Aggregate teacher-forced quality for a K/V precision sweep."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from aggregate_value_precision_sweep import (
    MATCHED_BIT_PAIRS,
    parse_config_bits,
    parse_seed_filter,
)
from spec_kv_statistics import bootstrap_mean_ci


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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


def paired_sequence_metric_differences(
    rows: List[Dict[str, str]],
    *,
    config_a: str,
    config_b: str,
    metric: str,
) -> List[float]:
    by_sequence: Dict[str, Dict[str, float]] = defaultdict(dict)
    for row in rows:
        candidate = row["candidate"]
        if candidate in {config_a, config_b}:
            by_sequence[row["sequence_idx"]][candidate] = float(row[metric])
    return [
        values[config_a] - values[config_b]
        for values in by_sequence.values()
        if config_a in values and config_b in values
    ]


def make_plot(rows: List[Dict[str, Any]], out_dir: Path) -> List[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    contexts = sorted({int(row["context"]) for row in rows})
    fig, axes = plt.subplots(1, len(contexts), figsize=(5.2 * len(contexts), 4.5), squeeze=False)
    for axis, context in zip(axes[0], contexts):
        subset = [row for row in rows if int(row["context"]) == context]
        for row in subset:
            cache_saved_fraction = float(
                row["cache_saved_fraction"]
                if "cache_saved_fraction" in row
                else row["draft_cache_saved_fraction"]
            )
            axis.errorbar(
                100.0 * cache_saved_fraction,
                float(row["kl_p_to_q_mean"]),
                yerr=[
                    [float(row["kl_p_to_q_mean"]) - float(row["kl_p_to_q_ci_low"])],
                    [float(row["kl_p_to_q_ci_high"]) - float(row["kl_p_to_q_mean"])],
                ],
                marker="o",
                capsize=3,
                color="#26456E" if int(row["k_bits"]) >= 16 else "#D1495B",
            )
            axis.annotate(
                str(row["config"]),
                (100.0 * cache_saved_fraction, float(row["kl_p_to_q_mean"])),
                fontsize=8,
            )
        axis.set_title(f"Context {context:,}")
        axis.set_xlabel("KV cache saved (%)")
        axis.set_ylabel("Teacher-forced KL (lower is better)")
        axis.grid(alpha=0.22)
    fig.suptitle("Ordinary LM Quality Under Matched K/V Precision", fontweight="bold")
    fig.tight_layout()
    paths = []
    for extension in ("png", "pdf"):
        path = out_dir / f"value_precision_quality.{extension}"
        fig.savefig(path, dpi=240, bbox_inches="tight")
        paths.append(str(path))
    plt.close(fig)
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep_dir", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument(
        "--seeds",
        default="",
        help="Optional comma-separated seed allowlist for provenance-safe partial aggregation.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    selected_seeds = parse_seed_filter(args.seeds)
    run_rows: List[Dict[str, Any]] = []
    sequence_metrics: Dict[Tuple[int, str], Dict[str, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    paired_metrics: Dict[Tuple[int, str, str], Dict[str, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    missing = []

    for seed_dir in sorted(args.sweep_dir.glob("ctx_*/seed_*")):
        seed_hint = int(seed_dir.name.removeprefix("seed_"))
        if selected_seeds is not None and seed_hint not in selected_seeds:
            continue
        summary_path = seed_dir / "summary.json"
        raw_path = seed_dir / "raw_sequence_rows.csv"
        if not summary_path.exists() or not raw_path.exists():
            missing.append(str(seed_dir))
            continue
        summary = read_json(summary_path)
        version = summary.get("runtime", {}).get("evaluator_version")
        if version != "teacher_forced_cached_v1":
            raise ValueError(f"Stale evaluator {version!r} in {summary_path}")
        context = int(summary["config"]["prompt_len"])
        seed = int(summary["config"]["seed"])
        if selected_seeds is not None and seed not in selected_seeds:
            continue
        raw_rows = read_csv(raw_path)
        for raw in raw_rows:
            name = raw["candidate"]
            if name == "none":
                continue
            for metric in ("kl_p_to_q", "delta_nll", "top1_match", "accept_mass"):
                sequence_metrics[(context, name)][metric].append(float(raw[metric]))
        for config_a, config_b, _ in MATCHED_BIT_PAIRS:
            for metric in ("kl_p_to_q", "delta_nll", "top1_match", "accept_mass"):
                paired_metrics[(context, config_a, config_b)][metric].extend(
                    paired_sequence_metric_differences(
                        raw_rows,
                        config_a=config_a,
                        config_b=config_b,
                        metric=metric,
                    )
                )
        for name, metrics in summary["summaries"].items():
            if name == "none":
                continue
            k_bits, v_bits = parse_config_bits(name)
            run_rows.append(
                {
                    "context": context,
                    "seed": seed,
                    "dataset_name": summary["config"]["dataset_name"],
                    "config": name,
                    "k_bits": k_bits,
                    "v_bits": v_bits,
                    "kl_p_to_q": metrics["kl_p_to_q"],
                    "delta_nll": metrics["delta_nll"],
                    "top1_match": metrics["top1_match"],
                    "accept_mass": metrics["accept_mass"],
                    "cache_saved_fraction": metrics["cache_saved_fraction"],
                    # Compatibility alias for speculative objective-comparison scripts.
                    "draft_cache_saved_fraction": metrics["cache_saved_fraction"],
                }
            )

    by_config: Dict[Tuple[int, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        by_config[(int(row["context"]), str(row["config"]))].append(row)

    grouped: List[Dict[str, Any]] = []
    for (context, name), values in sorted(by_config.items()):
        metrics = sequence_metrics[(context, name)]
        kl = bootstrap_mean_ci(metrics["kl_p_to_q"], seed=context + sum(map(ord, name)))
        nll = bootstrap_mean_ci(metrics["delta_nll"], seed=2 * context + sum(map(ord, name)))
        grouped.append(
            {
                "context": context,
                "config": name,
                "k_bits": values[0]["k_bits"],
                "v_bits": values[0]["v_bits"],
                "num_seeds": len(values),
                "paired_sequence_count": len(metrics["kl_p_to_q"]),
                "kl_p_to_q_mean": kl["mean"],
                "kl_p_to_q_ci_low": kl["ci_low"],
                "kl_p_to_q_ci_high": kl["ci_high"],
                "delta_nll_mean": nll["mean"],
                "delta_nll_ci_low": nll["ci_low"],
                "delta_nll_ci_high": nll["ci_high"],
                "top1_match_mean": statistics.mean(metrics["top1_match"]),
                "accept_mass_mean": statistics.mean(metrics["accept_mass"]),
                "cache_saved_fraction": statistics.mean(
                    float(row["cache_saved_fraction"]) for row in values
                ),
                "draft_cache_saved_fraction": statistics.mean(
                    float(row["cache_saved_fraction"]) for row in values
                ),
            }
        )

    if not grouped:
        raise ValueError("No complete teacher-forced value-precision outputs were found.")
    paired_comparisons: List[Dict[str, Any]] = []
    for context in sorted({int(row["context"]) for row in grouped}):
        for config_a, config_b, label in MATCHED_BIT_PAIRS:
            metrics = paired_metrics[(context, config_a, config_b)]
            if not metrics["kl_p_to_q"]:
                continue
            kl = bootstrap_mean_ci(
                metrics["kl_p_to_q"],
                seed=context * 10_000 + sum(map(ord, config_a + config_b)),
            )
            nll = bootstrap_mean_ci(
                metrics["delta_nll"],
                seed=context * 20_000 + sum(map(ord, config_a + config_b)),
            )
            paired_comparisons.append(
                {
                    "context": context,
                    "comparison": label,
                    "config_a": config_a,
                    "config_b": config_b,
                    "paired_sequence_count": len(metrics["kl_p_to_q"]),
                    "kl_contrast_mean": kl["mean"],
                    "kl_contrast_ci_low": kl["ci_low"],
                    "kl_contrast_ci_high": kl["ci_high"],
                    "delta_nll_contrast_mean": nll["mean"],
                    "delta_nll_contrast_ci_low": nll["ci_low"],
                    "delta_nll_contrast_ci_high": nll["ci_high"],
                    "top1_match_contrast_mean": statistics.mean(metrics["top1_match"]),
                    "accept_mass_contrast_mean": statistics.mean(metrics["accept_mass"]),
                }
            )
    write_csv(args.out_dir / "run_results.csv", run_rows)
    write_csv(args.out_dir / "grouped_results.csv", grouped)
    write_csv(args.out_dir / "paired_precision_contrasts.csv", paired_comparisons)
    payload = {
        "num_complete_runs": len({(row["context"], row["seed"]) for row in run_rows}),
        "missing_runs": missing,
        "selected_seeds": sorted(selected_seeds) if selected_seeds is not None else None,
        "grouped": grouped,
        "paired_precision_contrasts": paired_comparisons,
        "plots": make_plot(grouped, args.out_dir),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Aggregated {payload['num_complete_runs']} runs; missing {len(missing)}")
    print(args.out_dir / "summary.json")


if __name__ == "__main__":
    main()
