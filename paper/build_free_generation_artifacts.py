#!/usr/bin/env python3
"""Build provenance-checked cross-model free-generation paper artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EXPECTED_VERSION = "free_running_cached_v1"
CONFIGS = ("k4v8", "k8v4")
CONFIG_LABELS = {"k4v8": "K4V8", "k8v4": "K8V4"}
COLORS = {"k4v8": "#D1495B", "k8v4": "#17324D"}
METRICS = (
    ("exact_sequence_match", "Exact sequence"),
    ("token_match_fraction", "Token agreement"),
    ("prefix_retained_fraction", "Prefix retained"),
)


def parse_source(value: str) -> Tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label.strip() or not path.strip():
        raise argparse.ArgumentTypeError("Sources must use LABEL=PATH syntax.")
    return label.strip(), Path(path.strip())


def load_source(label: str, path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    runtime = payload.get("runtime", {})
    if runtime.get("source_evaluator_version") != EXPECTED_VERSION:
        raise ValueError(f"{path} does not aggregate {EXPECTED_VERSION!r} artifacts.")
    runs = payload.get("runs", [])
    if len(runs) < 3:
        raise ValueError(f"{path} has only {len(runs)} runs; at least three are required.")
    summaries = {row["config"]: row for row in payload.get("summaries", [])}
    missing = [config for config in CONFIGS if config not in summaries]
    if missing:
        raise ValueError(f"{path} is missing configurations: {missing}.")
    contrasts = {
        (row["left_config"], row["right_config"], row["metric"]): row
        for row in payload.get("paired_contrasts", [])
    }
    for metric, _ in METRICS:
        key = ("k8v4", "k4v8", metric)
        if key not in contrasts:
            raise ValueError(f"{path} is missing paired contrast {key}.")
    return {
        "label": label,
        "path": str(path),
        "runs": runs,
        "summaries": summaries,
        "contrasts": contrasts,
    }


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.frameon": False,
            "figure.dpi": 180,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )


def build_rows(sources: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    summaries: List[Dict[str, Any]] = []
    contrasts: List[Dict[str, Any]] = []
    for source in sources:
        for config in CONFIGS:
            row = source["summaries"][config]
            summaries.append(
                {
                    "model": source["label"],
                    "config": config,
                    "num_runs": row["num_runs"],
                    "num_prompt_occurrences": row["num_prompt_occurrences"],
                    "cache_saved_fraction": row["cache_saved_fraction"],
                    **{metric: row[metric] for metric, _ in METRICS},
                }
            )
        for metric, _ in METRICS:
            row = source["contrasts"][("k8v4", "k4v8", metric)]
            contrasts.append(
                {
                    "model": source["label"],
                    "metric": metric,
                    "num_paired_prompts": row["num_paired_prompts"],
                    "difference": row["difference"],
                    "ci_low": row["ci_low"],
                    "ci_high": row["ci_high"],
                }
            )
    return summaries, contrasts


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_table(path: Path, summaries: Sequence[Dict[str, Any]], contrasts: Sequence[Dict[str, Any]]) -> None:
    summary_map = {(row["model"], row["config"]): row for row in summaries}
    contrast_map = {(row["model"], row["metric"]): row for row in contrasts}
    models = list(dict.fromkeys(row["model"] for row in summaries))
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Model & K4V8 token (\%) & K8V4 token (\%) & $\Delta$ token (pp) & 95\% CI \\",
        r"\midrule",
    ]
    for model in models:
        left = summary_map[(model, "k4v8")]
        right = summary_map[(model, "k8v4")]
        contrast = contrast_map[(model, "token_match_fraction")]
        lines.append(
            f"{model} & {100 * left['token_match_fraction']:.2f} & "
            f"{100 * right['token_match_fraction']:.2f} & "
            f"{100 * contrast['difference']:+.2f} & "
            f"[{100 * contrast['ci_low']:+.2f}, {100 * contrast['ci_high']:+.2f}] \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot(out_dir: Path, summaries: Sequence[Dict[str, Any]], contrasts: Sequence[Dict[str, Any]]) -> None:
    models = list(dict.fromkeys(row["model"] for row in summaries))
    summary_map = {(row["model"], row["config"]): row for row in summaries}
    token_contrasts = {
        row["model"]: row for row in contrasts if row["metric"] == "token_match_fraction"
    }
    positions = list(range(len(models)))
    width = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.65), gridspec_kw={"width_ratios": [1.15, 1.0]})

    for offset, config in ((-width / 2, "k4v8"), (width / 2, "k8v4")):
        values = [100 * summary_map[(model, config)]["token_match_fraction"] for model in models]
        axes[0].bar(
            [position + offset for position in positions],
            values,
            width=width,
            color=COLORS[config],
            label=CONFIG_LABELS[config],
        )
    axes[0].set_xticks(positions, models)
    axes[0].set_ylabel("BF16 token agreement (%)")
    axes[0].set_ylim(0, 78)
    axes[0].set_title("Free-running continuation retention", fontweight="bold")
    axes[0].legend(ncol=2, loc="upper right")

    forest_y = list(reversed(positions))
    differences = [100 * token_contrasts[model]["difference"] for model in models]
    lower = [
        100 * (token_contrasts[model]["difference"] - token_contrasts[model]["ci_low"])
        for model in models
    ]
    upper = [
        100 * (token_contrasts[model]["ci_high"] - token_contrasts[model]["difference"])
        for model in models
    ]
    axes[1].errorbar(
        differences,
        forest_y,
        xerr=[lower, upper],
        fmt="o",
        markersize=4.5,
        capsize=3,
        color="#168C82",
        ecolor="#168C82",
        linewidth=1.4,
    )
    axes[1].axvline(0, color="#667085", linewidth=1, linestyle="--")
    axes[1].set_yticks(forest_y, models)
    axes[1].set_xlabel("K8V4 minus K4V8 token agreement (pp)")
    axes[1].set_title("Paired 95% bootstrap CI", fontweight="bold")

    for axis in axes:
        axis.grid(axis="y", color="#D9DDD8", linewidth=0.7, alpha=0.8)
        axis.set_axisbelow(True)
    fig.suptitle("Equal nominal K/V bits favor value precision under KIVI geometry", fontweight="bold")
    fig.tight_layout(w_pad=1.5)
    fig.savefig(out_dir / "free_generation_kv_asymmetry.pdf")
    fig.savefig(out_dir / "free_generation_kv_asymmetry.png", dpi=240)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", type=parse_source, required=True)
    parser.add_argument("--out_dir", type=Path, default=Path("paper/free_generation_artifacts"))
    args = parser.parse_args()

    sources = [load_source(label, path) for label, path in args.source]
    summaries, contrasts = build_rows(sources)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "free_generation_summary.csv", summaries)
    write_csv(args.out_dir / "free_generation_contrasts.csv", contrasts)
    write_table(args.out_dir / "free_generation_table.tex", summaries, contrasts)
    configure_style()
    plot(args.out_dir, summaries, contrasts)
    provenance = {
        "source_evaluator_version": EXPECTED_VERSION,
        "sources": [{"label": source["label"], "path": source["path"]} for source in sources],
        "num_models": len(sources),
        "num_runs": sum(len(source["runs"]) for source in sources),
    }
    (args.out_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote free-generation artifacts for {len(sources)} models to {args.out_dir}.")


if __name__ == "__main__":
    main()
