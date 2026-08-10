#!/usr/bin/env python3
"""Combine cached KV multiple-choice results across model families."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


EVALUATOR_VERSION = "kv_multiple_choice_cached_v2"
MODEL_LABELS = {
    "qwen25_15b": "Qwen2.5-1.5B",
    "llama32_3b": "Llama-3.2-3B",
    "olmo2_1b": "OLMo-2-1B",
    "smollm2_360m": "SmolLM2-360M",
}


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
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


def collect_summaries(
    sources: Iterable[Tuple[str, Path]],
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, str]],
]:
    grouped: List[Dict[str, Any]] = []
    comparisons: List[Dict[str, Any]] = []
    underfilled: List[Dict[str, Any]] = []
    rejected: List[Dict[str, str]] = []
    for model, path in sources:
        if not path.exists():
            rejected.append({"model": model, "path": str(path), "reason": "missing"})
            continue
        summary = read_json(path)
        if summary.get("evaluator_version") != EVALUATOR_VERSION:
            rejected.append(
                {
                    "model": model,
                    "path": str(path),
                    "reason": f"unexpected evaluator {summary.get('evaluator_version')!r}",
                }
            )
            continue
        model_label = MODEL_LABELS.get(model, model)
        grouped.extend(
            {
                "model": model,
                "model_label": model_label,
                "source_summary": str(path),
                **row,
            }
            for row in summary.get("grouped", [])
        )
        comparisons.extend(
            {
                "model": model,
                "model_label": model_label,
                "source_summary": str(path),
                **row,
            }
            for row in summary.get("comparisons", [])
        )
        underfilled.extend(
            {"model": model, "model_label": model_label, **row}
            for row in summary.get("underfilled_runs", [])
        )
    return grouped, comparisons, underfilled, rejected


def complete_plot_models(
    rows: Sequence[Dict[str, Any]],
    task: str,
    configs: Sequence[str],
    model_order: Sequence[str],
) -> List[str]:
    available = {
        (str(row["model"]), str(row["config"]))
        for row in rows
        if row["task"] == task
    }
    return [
        model
        for model in model_order
        if all((model, config) in available for config in configs)
    ]


def make_plot(rows: Sequence[Dict[str, Any]], out_dir: Path) -> List[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    configs = ["k8v4", "k4v8", "k4v4"]
    colors = {"k8v4": "#168C82", "k4v8": "#D1495B", "k4v4": "#667085"}
    tasks = sorted({str(row["task"]) for row in rows})
    models = [model for model in MODEL_LABELS if any(row["model"] == model for row in rows)]
    fig, axes = plt.subplots(1, len(tasks), figsize=(6.0 * len(tasks), 4.2), squeeze=False)
    width = 0.23
    for axis, task in zip(axes[0], tasks):
        lookup = {
            (str(row["model"]), str(row["config"])): row
            for row in rows
            if row["task"] == task
        }
        task_models = complete_plot_models(rows, task, configs, models)
        if not task_models:
            axis.set_visible(False)
            continue
        for config_idx, config in enumerate(configs):
            x = [
                index + (config_idx - 1) * width
                for index in range(len(task_models))
            ]
            selected = [lookup[(model, config)] for model in task_models]
            means = [100.0 * float(row["paired_delta_vs_bf16_mean"]) for row in selected]
            lows = [
                mean - 100.0 * float(row["paired_delta_vs_bf16_ci_low"])
                for mean, row in zip(means, selected)
            ]
            highs = [
                100.0 * float(row["paired_delta_vs_bf16_ci_high"]) - mean
                for mean, row in zip(means, selected)
            ]
            axis.bar(
                x,
                means,
                width,
                yerr=[lows, highs],
                capsize=2,
                color=colors[config],
                label=config.upper(),
            )
        axis.axhline(0.0, color="#18212B", linewidth=0.9)
        axis.set_xticks(
            range(len(task_models)),
            [
                MODEL_LABELS.get(model, model).replace("-", "-\n", 1)
                for model in task_models
            ],
        )
        axis.set_title(task.replace("_", " ").title())
        axis.set_ylabel("Accuracy change from BF16 (pp)")
        axis.grid(axis="y", alpha=0.2)
    axes[0][-1].legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("KV Quantization Task Accuracy Across Model Families", fontweight="bold")
    fig.tight_layout()
    paths = []
    for extension in ("png", "pdf"):
        path = out_dir / f"cross_family_task_accuracy.{extension}"
        fig.savefig(path, dpi=240, bbox_inches="tight")
        paths.append(str(path))
    plt.close(fig)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--qwen_summary", type=Path)
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sources = [
        (path.parents[1].name, path)
        for path in sorted(args.root.glob("*/aggregate/summary.json"))
    ]
    if args.qwen_summary is not None:
        sources.insert(0, ("qwen25_15b", args.qwen_summary))
    grouped, comparisons, underfilled, rejected = collect_summaries(sources)
    if not grouped:
        raise ValueError("No valid cross-family multiple-choice summaries were found.")

    write_csv(args.out_dir / "grouped_results.csv", grouped)
    write_csv(args.out_dir / "paired_comparisons.csv", comparisons)
    write_csv(args.out_dir / "underfilled_runs.csv", underfilled)
    write_csv(args.out_dir / "rejected_summaries.csv", rejected)
    payload = {
        "evaluator_version": EVALUATOR_VERSION,
        "models": sorted({str(row["model"]) for row in grouped}),
        "num_models": len({str(row["model"]) for row in grouped}),
        "num_grouped_rows": len(grouped),
        "num_comparisons": len(comparisons),
        "num_underfilled_runs": len(underfilled),
        "num_rejected_summaries": len(rejected),
        "underfilled_runs": underfilled,
        "rejected_summaries": rejected,
        "grouped": grouped,
        "comparisons": comparisons,
        "plots": make_plot(grouped, args.out_dir),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Aggregated {payload['num_models']} models; underfilled "
        f"{payload['num_underfilled_runs']}; rejected {payload['num_rejected_summaries']}"
    )
    print(args.out_dir / "summary.json")


if __name__ == "__main__":
    main()
