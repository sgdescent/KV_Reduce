#!/usr/bin/env python3
"""Compare per-token and KIVI-style per-channel key quantization."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from spec_kv_statistics import bootstrap_acceptance_contrast


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fields: List[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def exact_enough(row: Dict[str, str], tie_margin: float) -> bool:
    if float(row.get("matches_target_greedy", 0.0)) >= 0.5:
        return True
    try:
        margin = float(row.get("mismatch_min_top1_margin", "nan"))
    except (TypeError, ValueError):
        return False
    return math.isfinite(margin) and margin <= tie_margin


def load_runs(root: Path) -> Tuple[Dict[Tuple[int, int, str], Dict[str, Dict[str, str]]], Dict[Tuple[int, str], List[Dict[str, float]]]]:
    prompts: Dict[Tuple[int, int, str], Dict[str, Dict[str, str]]] = {}
    memory: Dict[Tuple[int, str], List[Dict[str, float]]] = defaultdict(list)
    for seed_dir in sorted(root.glob("ctx_*/seed_*")):
        summary_path = seed_dir / "summary.json"
        rows_path = seed_dir / "benchmark_rows.csv"
        if not summary_path.exists() or not rows_path.exists():
            continue
        summary = read_json(summary_path)
        if summary.get("runtime", {}).get("evaluator_version") != "cached_dynamic_v4":
            raise ValueError(f"Stale evaluator in {summary_path}")
        context = int(summary["config"]["prompt_len"])
        seed = int(summary["config"]["seed"])
        for row in read_csv(rows_path):
            prompts.setdefault((context, seed, row["prompt_idx"]), {})[row["config"]] = row
        for config, metrics in summary["summaries"].items():
            memory[(context, config)].append(
                {
                    "draft_cache_saved_fraction": float(metrics["draft_cache_saved_fraction"]),
                    "total_cache_saved_fraction": float(metrics["total_cache_saved_fraction"]),
                }
            )
    return prompts, memory


def parse_variant(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Variants must use label=/path/to/root.")
    label, path = value.split("=", 1)
    return label, Path(path)


def make_plot(rows: List[Dict[str, Any]], out_dir: Path) -> List[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    configs = sorted({str(row["config"]) for row in rows})
    variants = sorted({str(row["variant"]) for row in rows})
    contexts = sorted({int(row["context"]) for row in rows})
    fig, axes = plt.subplots(1, len(contexts), figsize=(6.0 * len(contexts), 4.5), squeeze=False)
    width = 0.8 / max(1, len(variants))
    for axis, context in zip(axes[0], contexts):
        for variant_idx, variant in enumerate(variants):
            subset = {
                str(row["config"]): row
                for row in rows
                if int(row["context"]) == context and str(row["variant"]) == variant
            }
            xs = [idx - 0.4 + width / 2 + variant_idx * width for idx in range(len(configs))]
            ys = [100.0 * float(subset[name]["axis_gain_mean"]) for name in configs]
            axis.bar(xs, ys, width=width, label=variant)
        axis.axhline(0.0, color="#222222", linewidth=1)
        axis.set_xticks(range(len(configs)), configs)
        axis.set_ylabel("Acceptance gain over per-token K (pp)")
        axis.set_title(f"Context {context:,}")
        axis.grid(axis="y", alpha=0.22)
    axes[0][-1].legend()
    fig.suptitle("Key Quantization Axis Ablation", fontweight="bold")
    fig.tight_layout()
    paths = []
    for extension in ("png", "pdf"):
        path = out_dir / f"key_axis_ablation.{extension}"
        fig.savefig(path, dpi=240, bbox_inches="tight")
        paths.append(str(path))
    plt.close(fig)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per_token_dir", required=True, type=Path)
    parser.add_argument("--variant", action="append", required=True, type=parse_variant)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--tie_margin", type=float, default=1e-3)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    baseline_prompts, _ = load_runs(args.per_token_dir)
    grouped_rows: List[Dict[str, Any]] = []
    for variant_label, variant_dir in args.variant:
        variant_prompts, variant_memory = load_runs(variant_dir)
        effects: Dict[
            Tuple[int, str], List[Tuple[Dict[str, str], Dict[str, str]]]
        ] = defaultdict(list)
        baseline_effects: Dict[
            Tuple[int, str], List[Tuple[Dict[str, str], Dict[str, str]]]
        ] = defaultdict(list)
        axis_gains: Dict[
            Tuple[int, str],
            List[Tuple[Dict[str, str], Dict[str, str], Dict[str, str], Dict[str, str]]],
        ] = defaultdict(list)
        for prompt_key in sorted(set(baseline_prompts) & set(variant_prompts)):
            context = prompt_key[0]
            baseline = baseline_prompts[prompt_key]
            variant = variant_prompts[prompt_key]
            for config in sorted((set(baseline) & set(variant)) - {"none"}):
                required = [baseline["none"], baseline[config], variant["none"], variant[config]]
                if not all(exact_enough(row, args.tie_margin) for row in required):
                    continue
                baseline_pair = (baseline[config], baseline["none"])
                variant_pair = (variant[config], variant["none"])
                baseline_effects[(context, config)].append(baseline_pair)
                effects[(context, config)].append(variant_pair)
                axis_gains[(context, config)].append(
                    (variant[config], variant["none"], baseline[config], baseline["none"])
                )

        for context, config in sorted(effects):
            variant_ci = bootstrap_acceptance_contrast(
                effects[(context, config)],
                (1.0, -1.0),
                seed=context + sum(map(ord, config + variant_label)),
            )
            baseline_ci = bootstrap_acceptance_contrast(
                baseline_effects[(context, config)],
                (1.0, -1.0),
                seed=context + sum(map(ord, config)),
            )
            gain_ci = bootstrap_acceptance_contrast(
                axis_gains[(context, config)],
                (1.0, -1.0, -1.0, 1.0),
                seed=context + sum(map(ord, variant_label)),
            )
            memory_rows = variant_memory[(context, config)]
            grouped_rows.append(
                {
                    "context": context,
                    "variant": variant_label,
                    "config": config,
                    "paired_prompt_count": len(effects[(context, config)]),
                    "per_token_effect_mean": baseline_ci["mean"],
                    "variant_effect_mean": variant_ci["mean"],
                    "variant_effect_ci_low": variant_ci["ci_low"],
                    "variant_effect_ci_high": variant_ci["ci_high"],
                    "axis_gain_mean": gain_ci["mean"],
                    "axis_gain_ci_low": gain_ci["ci_low"],
                    "axis_gain_ci_high": gain_ci["ci_high"],
                    "draft_cache_saved_fraction": statistics.mean(
                        row["draft_cache_saved_fraction"] for row in memory_rows
                    ),
                    "total_cache_saved_fraction": statistics.mean(
                        row["total_cache_saved_fraction"] for row in memory_rows
                    ),
                }
            )

    if not grouped_rows:
        raise ValueError("No matched axis-ablation prompts were found.")
    write_csv(args.out_dir / "grouped_results.csv", grouped_rows)
    payload = {"grouped": grouped_rows, "plots": make_plot(grouped_rows, args.out_dir)}
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(args.out_dir / "summary.json")


if __name__ == "__main__":
    main()
