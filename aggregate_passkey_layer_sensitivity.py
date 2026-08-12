#!/usr/bin/env python3
"""Aggregate paired one-layer K/V passkey sensitivity experiments."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

from spec_kv_statistics import bootstrap_mean_ci


EVALUATOR_VERSION = "kv_multiple_choice_cached_v2"
GENERATOR_VERSION = "synthetic_associative_passkey_v3"


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def paired_component_effects(
    rows: Sequence[Dict[str, str]],
    *,
    baseline: str,
    k_config: str,
    v_config: str,
    metric: str = "raw_correct",
) -> Dict[str, List[float]]:
    by_example: Dict[tuple[str, str], Dict[str, float]] = defaultdict(dict)
    expected = {baseline, k_config, v_config}
    for row in rows:
        if row["config"] not in expected:
            continue
        by_example[(row["seed"], row["source_idx"])][row["config"]] = float(row[metric])
    complete = [values for values in by_example.values() if expected.issubset(values)]
    return {
        "k_harm": [values[baseline] - values[k_config] for values in complete],
        "v_harm": [values[baseline] - values[v_config] for values in complete],
        "k_minus_v_harm": [values[v_config] - values[k_config] for values in complete],
    }


def make_plot(rows: Sequence[Dict[str, Any]], out_dir: Path) -> List[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    layers = [int(row["layer"]) for row in rows]
    k_harm = [100.0 * float(row["k_harm_mean"]) for row in rows]
    v_harm = [100.0 * float(row["v_harm_mean"]) for row in rows]
    positions = list(range(len(layers)))
    width = 0.38
    fig, axis = plt.subplots(figsize=(8.2, 4.7))
    axis.bar([p - width / 2 for p in positions], k_harm, width, label="Lower K: 4 to 2 bits")
    axis.bar([p + width / 2 for p in positions], v_harm, width, label="Lower V: 4 to 2 bits")
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(positions, [str(layer) for layer in layers])
    axis.set_xlabel("Model layer")
    axis.set_ylabel("Passkey accuracy harm (percentage points)")
    axis.set_title("Layer-wise Retrieval Sensitivity", fontweight="bold")
    axis.grid(axis="y", alpha=0.22)
    axis.legend(frameon=False)
    fig.tight_layout()
    paths = []
    for extension in ("png", "pdf"):
        path = out_dir / f"passkey_layer_sensitivity.{extension}"
        fig.savefig(path, dpi=240, bbox_inches="tight")
        paths.append(str(path))
    plt.close(fig)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--context", type=int, default=16384)
    parser.add_argument("--expected_seeds", default="0,1,2")
    parser.add_argument("--expected_examples_per_run", type=int, default=32)
    parser.add_argument("--minimum_source_index", type=int, default=192)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--require_complete", action="store_true")
    parser.add_argument("--bootstrap_samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((args.root / "allocations" / "manifest.json").read_text(encoding="utf-8"))
    expected_seeds = [int(item.strip()) for item in args.expected_seeds.split(",") if item.strip()]
    all_rows: List[Dict[str, str]] = []
    missing = []
    underfilled = []
    run_source_ranges = []
    expected_configs = {"none", manifest["baseline"]} | {
        candidate["name"] for candidate in manifest["candidates"]
    }
    for seed in expected_seeds:
        run_dir = args.root / f"ctx_{args.context}" / f"seed_{seed}"
        summary_path = run_dir / "summary.json"
        rows_path = run_dir / "example_rows.csv"
        if not summary_path.exists() or not rows_path.exists():
            missing.append(str(run_dir))
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        runtime = summary.get("runtime", {})
        if runtime.get("evaluator_version") != EVALUATOR_VERSION:
            raise ValueError(f"Unexpected evaluator in {summary_path}.")
        if runtime.get("task_generator_version") != GENERATOR_VERSION:
            raise ValueError(f"Unexpected generator in {summary_path}.")
        if summary.get("task") != "passkey" or summary.get("primary_metric") != "raw_accuracy":
            raise ValueError(f"Unexpected task or metric in {summary_path}.")
        config = summary.get("config", {})
        if int(config.get("max_prompt_tokens", -1)) != args.context:
            raise ValueError(f"Context mismatch in {summary_path}.")
        if int(config.get("passkey_num_choices", -1)) != 16:
            raise ValueError(f"Choice-count mismatch in {summary_path}.")
        if str(config.get("passkey_variant")) != "confusable_records":
            raise ValueError(f"Passkey-variant mismatch in {summary_path}.")
        if str(runtime.get("key_quant_axis")) != "per_channel":
            raise ValueError(f"Key-quantization-axis mismatch in {summary_path}.")
        if int(summary.get("num_examples", -1)) != args.expected_examples_per_run:
            underfilled.append(str(run_dir))
            continue
        observed_configs = set(summary.get("summaries", {}))
        if observed_configs != expected_configs:
            raise ValueError(
                f"Configuration mismatch in {summary_path}: "
                f"missing={sorted(expected_configs - observed_configs)}, "
                f"extra={sorted(observed_configs - expected_configs)}"
            )
        run_rows = read_csv(rows_path)
        expected_run_rows = args.expected_examples_per_run * len(expected_configs)
        if len(run_rows) != expected_run_rows:
            underfilled.append(str(run_dir))
            continue
        unique_rows = {(row["seed"], row["source_idx"], row["config"]) for row in run_rows}
        if len(unique_rows) != len(run_rows):
            raise ValueError(f"Duplicate example/config rows in {rows_path}.")
        all_rows.extend(run_rows)
        source_range = tuple(int(value) for value in summary["source_index_range"])
        if source_range[0] < args.minimum_source_index:
            raise ValueError(
                f"Source range {source_range} overlaps the reserved powered split "
                f"ending before {args.minimum_source_index}."
            )
        run_source_ranges.append(source_range)
    if args.require_complete and (missing or underfilled):
        raise ValueError(f"Incomplete sweep: missing={missing}, underfilled={underfilled}")
    if not all_rows:
        raise ValueError("No complete passkey layer-sensitivity runs were found.")
    if len(set(run_source_ranges)) != len(run_source_ranges):
        raise ValueError(f"Passkey seed shards overlap: {run_source_ranges}")

    candidate_by_layer = defaultdict(dict)
    for candidate in manifest["candidates"]:
        candidate_by_layer[int(candidate["layer"])][candidate["component"]] = candidate["name"]
    layer_rows = []
    for layer in manifest["selected_layers"]:
        configs = candidate_by_layer[int(layer)]
        if set(configs) != {"k", "v"}:
            raise ValueError(f"Layer {layer} is missing a K or V candidate.")
        effects = paired_component_effects(
            all_rows,
            baseline=manifest["baseline"],
            k_config=configs["k"],
            v_config=configs["v"],
        )
        stats = {
            name: bootstrap_mean_ci(
                values,
                seed=args.seed + 100 * int(layer) + offset,
                samples=args.bootstrap_samples,
            )
            for offset, (name, values) in enumerate(effects.items())
        }
        layer_rows.append(
            {
                "layer": int(layer),
                "paired_examples": len(effects["k_harm"]),
                **{
                    f"{name}_{field}": value
                    for name, result in stats.items()
                    for field, value in result.items()
                },
            }
        )

    plot_paths = make_plot(layer_rows, args.out_dir)
    complete_gate = not missing and not underfilled and len(all_rows) == (
        len(expected_seeds) * args.expected_examples_per_run * len(expected_configs)
    )
    payload = {
        "evaluator_version": EVALUATOR_VERSION,
        "generator_version": GENERATOR_VERSION,
        "context": args.context,
        "expected_seeds": expected_seeds,
        "expected_examples_per_run": args.expected_examples_per_run,
        "minimum_source_index": args.minimum_source_index,
        "baseline": manifest["baseline"],
        "selected_layers": manifest["selected_layers"],
        "complete_gate": complete_gate,
        "missing": missing,
        "underfilled": underfilled,
        "source_index_ranges": run_source_ranges,
        "layer_results": layer_rows,
        "plots": plot_paths,
    }
    write_csv(args.out_dir / "layer_results.csv", layer_rows)
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.require_complete and not complete_gate:
        raise ValueError("Passkey layer-sensitivity completeness gate failed.")
    print(args.out_dir / "summary.json")


if __name__ == "__main__":
    main()
