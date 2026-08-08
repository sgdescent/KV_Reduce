#!/usr/bin/env python3
"""Aggregate cache-resident SpecKV campaign outputs into paper artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


PAIR_LABELS = {
    "qwen25_3b_15b": "Qwen2.5 3B/1.5B",
    "qwen25_7b_3b": "Qwen2.5 7B/3B",
    "qwen3_8b_4b": "Qwen3 8B/4B",
    "llama31_8b_llama32_3b": "Llama 3.1 8B/3.2 3B",
    "olmo2_7b_1b": "OLMo-2 7B/1B",
    "smollm2_17b_360m": "SmolLM2 1.7B/360M",
}

CONFIG_LABELS = {
    "none": "FP16",
    "k8v8": "K8 V8",
    "k8v4": "K8 V4",
    "k4v8": "K4 V8",
    "k4v4": "K4 V4",
    "sensitivity_aware": "Sensitivity-aware",
}

COLORS = {
    "none": "#17324D",
    "k8v8": "#D99A2B",
    "k8v4": "#168C82",
    "k4v8": "#D1495B",
    "k4v4": "#667085",
    "sensitivity_aware": "#6E56CF",
}


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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


def context_from_label(label: str, summary: Dict[str, Any]) -> int:
    config = summary.get("config", {})
    if "prompt_len" in config:
        return int(config["prompt_len"])
    marker = "ctx"
    if marker in label:
        suffix = label.split(marker, 1)[1].split("_", 1)[0]
        return int(suffix)
    return -1


def valid_cached_summary(summary: Dict[str, Any]) -> Tuple[bool, str]:
    runtime = summary.get("runtime", {})
    if runtime.get("evaluator_version") != "cached_dynamic_v2":
        return False, "unsupported or missing evaluator version"
    if not runtime.get("target_cache_reused"):
        return False, "target cache was not reused"
    if not runtime.get("draft_cache_reused"):
        return False, "draft cache was not reused"
    if runtime.get("target_reference_generation_in_timing") is not False:
        return False, "target reference generation timing is ambiguous"
    return True, ""


def acceptance_ratio(rows: Sequence[Dict[str, str]], indices: Iterable[int] | None = None) -> float:
    selected = rows if indices is None else [rows[index] for index in indices]
    proposed = sum(float(row["proposed_tokens"]) for row in selected)
    accepted = sum(float(row["accepted_tokens"]) for row in selected)
    return accepted / proposed if proposed > 0 else 0.0


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return float("nan")
    return float(np.quantile(np.asarray(values, dtype=np.float64), quantile))


def bootstrap_ratio_ci(
    rows: Sequence[Dict[str, str]],
    *,
    samples: int,
    seed: int,
) -> Tuple[float, float]:
    if len(rows) < 2 or samples <= 0:
        value = acceptance_ratio(rows)
        return value, value
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        indices = [rng.randrange(len(rows)) for _ in rows]
        estimates.append(acceptance_ratio(rows, indices))
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def paired_bootstrap_delta_ci(
    left_rows: Sequence[Dict[str, str]],
    right_rows: Sequence[Dict[str, str]],
    *,
    samples: int,
    seed: int,
) -> Tuple[float, float]:
    left = {int(row["prompt_idx"]): row for row in left_rows}
    right = {int(row["prompt_idx"]): row for row in right_rows}
    prompt_ids = sorted(set(left) & set(right))
    if not prompt_ids:
        return float("nan"), float("nan")
    if len(prompt_ids) < 2 or samples <= 0:
        delta = acceptance_ratio([left[idx] for idx in prompt_ids]) - acceptance_ratio(
            [right[idx] for idx in prompt_ids]
        )
        return delta, delta
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        sampled_ids = [prompt_ids[rng.randrange(len(prompt_ids))] for _ in prompt_ids]
        estimates.append(
            acceptance_ratio([left[idx] for idx in sampled_ids])
            - acceptance_ratio([right[idx] for idx in sampled_ids])
        )
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.edgecolor": "#18212B",
            "axes.labelcolor": "#18212B",
            "xtick.color": "#18212B",
            "ytick.color": "#18212B",
            "text.color": "#18212B",
            "legend.frameon": False,
            "figure.dpi": 180,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )


def collect_campaign(
    results_root: Path,
    *,
    bootstrap_samples: int,
    seed: int,
    tie_tolerance: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, str]]]:
    metrics: List[Dict[str, Any]] = []
    comparisons: List[Dict[str, Any]] = []
    rejected: List[Dict[str, str]] = []

    for summary_path in sorted(results_root.glob("*/*/summary.json")):
        pair_name = summary_path.parents[1].name
        run_label = summary_path.parent.name
        if run_label == "smoke" or run_label.startswith("sensitivity_"):
            continue
        summary = read_json(summary_path)
        valid, reason = valid_cached_summary(summary)
        if not valid:
            rejected.append({"path": str(summary_path), "reason": reason})
            continue
        rows_path = summary_path.parent / "benchmark_rows.csv"
        if not rows_path.exists():
            rejected.append({"path": str(summary_path), "reason": "missing benchmark_rows.csv"})
            continue

        rows_by_config: Dict[str, List[Dict[str, str]]] = defaultdict(list)
        for row in read_csv(rows_path):
            rows_by_config[row["config"]].append(row)
        context = context_from_label(run_label, summary)
        baseline = summary.get("summaries", {}).get("none", {})
        baseline_accept = float(baseline.get("overall_accept_rate", 0.0))

        for config_name, config_summary in summary.get("summaries", {}).items():
            prompt_rows = rows_by_config.get(config_name, [])
            mismatch_rows = [row for row in prompt_rows if float(row.get("matches_target_greedy", 0.0)) < 1.0]
            tie_mismatches = 0
            non_tie_mismatches = 0
            unknown_mismatches = 0
            for row in mismatch_rows:
                try:
                    margin = float(row["mismatch_target_top1_margin"])
                except (KeyError, TypeError, ValueError):
                    unknown_mismatches += 1
                    continue
                if math.isnan(margin):
                    unknown_mismatches += 1
                elif margin <= tie_tolerance:
                    tie_mismatches += 1
                else:
                    non_tie_mismatches += 1
            ci_low, ci_high = bootstrap_ratio_ci(
                prompt_rows,
                samples=bootstrap_samples,
                seed=seed + len(metrics),
            )
            exact_match = float(config_summary.get("matches_target_greedy", 0.0))
            metrics.append(
                {
                    "pair": pair_name,
                    "pair_label": PAIR_LABELS.get(pair_name, pair_name),
                    "run": run_label,
                    "context": context,
                    "dataset": summary.get("config", {}).get("dataset_name", ""),
                    "seed": int(summary.get("config", {}).get("seed", 0)),
                    "config": config_name,
                    "config_label": CONFIG_LABELS.get(config_name, config_name),
                    "num_prompts": int(summary.get("num_prompts", len(prompt_rows))),
                    "accept_rate": float(config_summary.get("overall_accept_rate", 0.0)),
                    "accept_ci_low": ci_low,
                    "accept_ci_high": ci_high,
                    "acceptance_retained": (
                        float(config_summary.get("overall_accept_rate", 0.0)) / baseline_accept
                        if baseline_accept > 0
                        else 0.0
                    ),
                    "accepted_per_round": float(config_summary.get("accepted_per_round", 0.0)),
                    "top1_match": float(config_summary.get("round_top1_match", 0.0)),
                    "round_js": float(config_summary.get("round_js", 0.0)),
                    "accept_mass": float(config_summary.get("round_accept_mass", 0.0)),
                    "draft_cache_saved_fraction": float(
                        config_summary.get("draft_cache_saved_fraction", 0.0)
                    ),
                    "total_cache_saved_fraction": float(
                        config_summary.get("total_cache_saved_fraction", 0.0)
                    ),
                    "exact_match": exact_match,
                    "tie_consistent_mismatches": tie_mismatches,
                    "non_tie_mismatches": non_tie_mismatches,
                    "unknown_mismatches": unknown_mismatches,
                    "valid_exact_generation": non_tie_mismatches == 0 and unknown_mismatches == 0,
                    "summary_path": str(summary_path),
                }
            )

        if "k8v4" in rows_by_config and "k4v8" in rows_by_config:
            left_rate = acceptance_ratio(rows_by_config["k8v4"])
            right_rate = acceptance_ratio(rows_by_config["k4v8"])
            delta_low, delta_high = paired_bootstrap_delta_ci(
                rows_by_config["k8v4"],
                rows_by_config["k4v8"],
                samples=bootstrap_samples,
                seed=seed + len(comparisons) + 10000,
            )
            memory = summary["summaries"]["k8v4"]
            comparisons.append(
                {
                    "pair": pair_name,
                    "pair_label": PAIR_LABELS.get(pair_name, pair_name),
                    "run": run_label,
                    "context": context,
                    "k8v4_accept_rate": left_rate,
                    "k4v8_accept_rate": right_rate,
                    "accept_rate_delta": left_rate - right_rate,
                    "delta_ci_low": delta_low,
                    "delta_ci_high": delta_high,
                    "k8v4_wins": left_rate > right_rate,
                    "total_cache_saved_fraction": float(memory["total_cache_saved_fraction"]),
                }
            )

    return metrics, comparisons, rejected


def plot_cross_family(metrics: Sequence[Dict[str, Any]], out_dir: Path) -> None:
    selected = [
        row
        for row in metrics
        if row["context"] == 1024
        and row["dataset"] == "wikitext"
        and row["config"] in {"none", "k8v4", "k4v8"}
    ]
    pairs = sorted({row["pair"] for row in selected}, key=lambda pair: list(PAIR_LABELS).index(pair))
    if not pairs:
        return
    lookup = {(row["pair"], row["config"]): row for row in selected}
    configs = ["none", "k8v4", "k4v8"]
    x = np.arange(len(pairs), dtype=np.float64)
    width = 0.25
    fig, ax = plt.subplots(figsize=(7.2, 2.75))
    for config_idx, config_name in enumerate(configs):
        values = [lookup.get((pair, config_name), {}).get("accept_rate", np.nan) for pair in pairs]
        ax.bar(
            x + (config_idx - 1) * width,
            values,
            width,
            label=CONFIG_LABELS[config_name],
            color=COLORS[config_name],
        )
    ax.set_xticks(x, [PAIR_LABELS[pair].replace("/", "/\n") for pair in pairs])
    ax.set_ylabel("Acceptance rate")
    ax.set_ylim(0, 0.75)
    ax.grid(axis="y", color="#D9DDD8", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    ax.set_title("Equal-memory K/V precision across model families", pad=18, fontweight="bold")
    fig.savefig(out_dir / "cross_family_equal_memory.pdf")
    fig.savefig(out_dir / "cross_family_equal_memory.png")
    plt.close(fig)


def plot_pareto(metrics: Sequence[Dict[str, Any]], out_dir: Path) -> None:
    selected = [
        row
        for row in metrics
        if row["context"] == 1024
        and row["dataset"] == "wikitext"
        and row["config"] in {"k8v8", "k8v4", "k4v8", "k4v4", "sensitivity_aware"}
    ]
    if not selected:
        return
    fig, ax = plt.subplots(figsize=(5.1, 3.0))
    for config_name in ["k8v8", "k8v4", "k4v8", "k4v4", "sensitivity_aware"]:
        rows = [row for row in selected if row["config"] == config_name]
        if not rows:
            continue
        ax.scatter(
            [100.0 * row["total_cache_saved_fraction"] for row in rows],
            [100.0 * row["acceptance_retained"] for row in rows],
            s=40,
            alpha=0.85,
            label=CONFIG_LABELS[config_name],
            color=COLORS[config_name],
            edgecolor="white",
            linewidth=0.5,
        )
    ax.axhline(100.0, color="#18212B", linestyle="--", linewidth=0.8)
    ax.axhline(90.0, color="#667085", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Total target + draft KV memory saved (%)")
    ax.set_ylabel("Native acceptance retained (%)")
    ax.grid(color="#D9DDD8", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(ncol=2, fontsize=8)
    ax.set_title("Cross-family memory/acceptance trade-off", fontweight="bold")
    fig.savefig(out_dir / "campaign_memory_acceptance_pareto.pdf")
    fig.savefig(out_dir / "campaign_memory_acceptance_pareto.png")
    plt.close(fig)


def plot_sensitivity_heatmap(results_root: Path, out_dir: Path) -> None:
    rows: List[Tuple[str, Dict[int, float]]] = []
    for profile_path in sorted(results_root.glob("*/sensitivity_top8_ctx1024/profile_summary.csv")):
        pair = profile_path.parents[1].name
        component_rows: Dict[Tuple[int, str], float] = {}
        for row in read_csv(profile_path):
            if row.get("component") not in {"k", "v"} or int(row.get("bits", -1)) != 4:
                continue
            component_rows[(int(row["layer"]), row["component"])] = float(row["accept_rate_drop"])
        layers = sorted({layer for layer, _ in component_rows})
        differences = {
            layer: component_rows.get((layer, "k"), 0.0) - component_rows.get((layer, "v"), 0.0)
            for layer in layers
        }
        if differences:
            rows.append((pair, differences))
    if not rows:
        return

    width = max(len(values) for _, values in rows)
    matrix = np.full((len(rows), width), np.nan, dtype=np.float64)
    for row_idx, (_, values) in enumerate(rows):
        ordered = [values[layer] for layer in sorted(values)]
        matrix[row_idx, width - len(ordered) :] = ordered
    limit = max(0.01, float(np.nanmax(np.abs(matrix))))
    fig, ax = plt.subplots(figsize=(6.0, 0.55 * len(rows) + 1.4))
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_yticks(np.arange(len(rows)), [PAIR_LABELS.get(pair, pair) for pair, _ in rows])
    ax.set_xticks(np.arange(width), [f"top-{width - index}" for index in range(width)])
    ax.set_xlabel("Draft layer position")
    ax.set_title("4-bit sensitivity: acceptance drop(K) - drop(V)", fontweight="bold")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    colorbar.set_label("Positive means keys are more sensitive")
    fig.savefig(out_dir / "layerwise_k_minus_v_sensitivity.pdf")
    fig.savefig(out_dir / "layerwise_k_minus_v_sensitivity.png")
    plt.close(fig)


def write_latex_table(comparisons: Sequence[Dict[str, Any]], out_dir: Path) -> None:
    rows = [
        row
        for row in comparisons
        if row["context"] == 1024 and row["run"] == "wikitext_ctx1024"
    ]
    rows.sort(key=lambda row: list(PAIR_LABELS).index(row["pair"]))
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Target / draft & K8 V4 & K4 V8 & $\Delta$ accept. & Total KV saved \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['pair_label']} & {row['k8v4_accept_rate']:.3f} & "
            f"{row['k4v8_accept_rate']:.3f} & {row['accept_rate_delta']:+.3f} & "
            f"{100.0 * row['total_cache_saved_fraction']:.1f}\\% \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (out_dir / "campaign_equal_memory_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_root", type=Path, default=Path("outputs/iclr_spec_kv"))
    parser.add_argument("--out_dir", type=Path, default=Path("paper/campaign_artifacts"))
    parser.add_argument("--bootstrap_samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument(
        "--tie_tolerance",
        type=float,
        default=1e-3,
        help="Treat greedy mismatches at or below this target top-1 margin as numerical ties.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    configure_plot_style()
    metrics, comparisons, rejected = collect_campaign(
        args.results_root,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        tie_tolerance=args.tie_tolerance,
    )
    write_csv(args.out_dir / "campaign_metrics.csv", metrics)
    write_csv(args.out_dir / "equal_memory_comparisons.csv", comparisons)
    write_csv(args.out_dir / "rejected_artifacts.csv", rejected)
    plot_cross_family(metrics, args.out_dir)
    plot_pareto(metrics, args.out_dir)
    plot_sensitivity_heatmap(args.results_root, args.out_dir)
    write_latex_table(comparisons, args.out_dir)

    aggregate = {
        "results_root": str(args.results_root),
        "tie_tolerance": args.tie_tolerance,
        "num_metric_rows": len(metrics),
        "num_equal_memory_comparisons": len(comparisons),
        "num_rejected_artifacts": len(rejected),
        "all_exact": all(row["valid_exact_generation"] for row in metrics) if metrics else False,
        "k8v4_wins": sum(bool(row["k8v4_wins"]) for row in comparisons),
        "comparisons": comparisons,
        "rejected_artifacts": rejected,
    }
    (args.out_dir / "campaign_aggregate.json").write_text(
        json.dumps(aggregate, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Aggregated {len(metrics)} metric rows and {len(comparisons)} equal-memory comparisons; "
        f"rejected {len(rejected)} stale/incomplete artifacts."
    )
    if comparisons:
        print(
            f"K8V4 wins: {aggregate['k8v4_wins']}/{len(comparisons)}; "
            f"all exact: {aggregate['all_exact']}"
        )
    print(f"Artifacts: {args.out_dir}")


if __name__ == "__main__":
    main()
