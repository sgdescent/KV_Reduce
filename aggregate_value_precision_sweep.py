#!/usr/bin/env python3
"""Aggregate a fixed-K, variable-V speculative KV quantization sweep."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from spec_kv_statistics import bootstrap_acceptance_contrast


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


def parse_config_bits(name: str) -> Tuple[int, int]:
    if name == "none":
        return 16, 16
    match = re.fullmatch(r"k(\d+)v(\d+)", name)
    if not match:
        raise ValueError(f"Unsupported sweep config name: {name}")
    return int(match.group(1)), int(match.group(2))


def exactness_status(row: Dict[str, str], *, tie_margin: float) -> str:
    if float(row.get("matches_target_greedy", 0.0)) >= 0.5:
        return "exact"
    try:
        margin = float(row.get("mismatch_min_top1_margin", "nan"))
    except (TypeError, ValueError):
        margin = float("nan")
    if math.isfinite(margin) and margin <= tie_margin:
        return "numerical_tie"
    return "non_tie_or_unknown"


def aggregate_prompt_effects(
    rows: Iterable[Dict[str, str]],
    *,
    configs: Iterable[str],
    tie_margin: float,
) -> Tuple[Dict[str, List[Tuple[Dict[str, str], Dict[str, str]]]], Counter[str], int]:
    expected = set(configs) | {"none"}
    by_prompt: Dict[str, Dict[str, Dict[str, str]]] = defaultdict(dict)
    invalid_prompts = set()
    counts: Counter[str] = Counter()
    for row in rows:
        config = row.get("config", "")
        if config not in expected:
            continue
        prompt = row["prompt_idx"]
        by_prompt[prompt][config] = row
        status = exactness_status(row, tie_margin=tie_margin)
        counts[status] += 1
        if status == "non_tie_or_unknown":
            invalid_prompts.add(prompt)

    effects: Dict[str, List[Tuple[Dict[str, str], Dict[str, str]]]] = defaultdict(list)
    for prompt, values in by_prompt.items():
        if prompt in invalid_prompts or "none" not in values:
            continue
        for config in expected - {"none"}:
            if config in values:
                effects[config].append((values[config], values["none"]))
    return effects, counts, len(invalid_prompts)


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
            axis.errorbar(
                100.0 * float(row["total_cache_saved_fraction"]),
                100.0 * float(row["paired_acceptance_delta_mean"]),
                yerr=[
                    [100.0 * (float(row["paired_acceptance_delta_mean"]) - float(row["paired_acceptance_delta_ci_low"]))],
                    [100.0 * (float(row["paired_acceptance_delta_ci_high"]) - float(row["paired_acceptance_delta_mean"]))],
                ],
                marker="o",
                capsize=3,
                color="#26456E" if int(row["k_bits"]) >= 16 else "#D1495B",
            )
            axis.annotate(str(row["config"]), (100.0 * float(row["total_cache_saved_fraction"]), 100.0 * float(row["paired_acceptance_delta_mean"])), fontsize=8)
        axis.axhline(0.0, color="#222222", linewidth=1)
        axis.set_title(f"Context {context:,}")
        axis.set_xlabel("Total target+draft KV saved (%)")
        axis.set_ylabel("Acceptance change (pp)")
        axis.grid(alpha=0.22)
    fig.suptitle("Value Precision Sweep with Key Precision Preserved", fontweight="bold")
    fig.tight_layout()
    paths = []
    for extension in ("png", "pdf"):
        path = out_dir / f"value_precision_pareto.{extension}"
        fig.savefig(path, dpi=240, bbox_inches="tight")
        paths.append(str(path))
    plt.close(fig)
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep_dir", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--exactness_tie_margin", type=float, default=1e-3)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_rows: List[Dict[str, Any]] = []
    prompt_effects: Dict[
        Tuple[int, str], List[Tuple[Dict[str, str], Dict[str, str]]]
    ] = defaultdict(list)
    exactness: Counter[str] = Counter()
    invalid_prompts = 0
    missing = []

    for seed_dir in sorted(args.sweep_dir.glob("ctx_*/seed_*")):
        summary_path = seed_dir / "summary.json"
        benchmark_path = seed_dir / "benchmark_rows.csv"
        if not summary_path.exists() or not benchmark_path.exists():
            missing.append(str(seed_dir))
            continue
        summary = read_json(summary_path)
        version = summary.get("runtime", {}).get("evaluator_version")
        if version != "cached_dynamic_v4":
            raise ValueError(f"Stale evaluator {version!r} in {summary_path}")
        config = summary["config"]
        context = int(config["prompt_len"])
        seed = int(config["seed"])
        names = [name for name in summary["quant_configs"] if name != "none"]
        effects, counts, invalid = aggregate_prompt_effects(
            read_csv(benchmark_path),
            configs=names,
            tie_margin=args.exactness_tie_margin,
        )
        exactness.update(counts)
        invalid_prompts += invalid
        for name, values in effects.items():
            prompt_effects[(context, name)].extend(values)
        for name in names:
            metrics = summary["summaries"][name]
            k_bits, v_bits = parse_config_bits(name)
            run_rows.append(
                {
                    "context": context,
                    "seed": seed,
                    "dataset_name": config["dataset_name"],
                    "config": name,
                    "k_bits": k_bits,
                    "v_bits": v_bits,
                    "accept_rate": metrics["overall_accept_rate"],
                    "round_js": metrics["round_js"],
                    "round_accept_mass": metrics["round_accept_mass"],
                    "round_top1_match": metrics["round_top1_match"],
                    "draft_cache_saved_fraction": metrics["draft_cache_saved_fraction"],
                    "total_cache_saved_fraction": metrics["total_cache_saved_fraction"],
                }
            )

    grouped: List[Dict[str, Any]] = []
    by_config: Dict[Tuple[int, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        by_config[(int(row["context"]), str(row["config"]))].append(row)
    for (context, name), values in sorted(by_config.items()):
        effect = bootstrap_acceptance_contrast(
            prompt_effects[(context, name)],
            (1.0, -1.0),
            seed=context * 100 + sum(ord(char) for char in name),
        )
        grouped.append(
            {
                "context": context,
                "config": name,
                "k_bits": values[0]["k_bits"],
                "v_bits": values[0]["v_bits"],
                "num_seeds": len(values),
                "paired_prompt_count": len(prompt_effects[(context, name)]),
                "paired_acceptance_delta_mean": effect["mean"],
                "paired_acceptance_delta_ci_low": effect["ci_low"],
                "paired_acceptance_delta_ci_high": effect["ci_high"],
                "round_js_mean": statistics.mean(float(row["round_js"]) for row in values),
                "round_accept_mass_mean": statistics.mean(float(row["round_accept_mass"]) for row in values),
                "draft_cache_saved_fraction": statistics.mean(float(row["draft_cache_saved_fraction"]) for row in values),
                "total_cache_saved_fraction": statistics.mean(float(row["total_cache_saved_fraction"]) for row in values),
            }
        )

    if not grouped:
        raise ValueError("No complete value-precision sweep outputs were found.")
    write_csv(args.out_dir / "run_results.csv", run_rows)
    write_csv(args.out_dir / "grouped_results.csv", grouped)
    payload = {
        "num_complete_runs": len({(row["context"], row["seed"]) for row in run_rows}),
        "missing_runs": missing,
        "exactness_tie_margin": args.exactness_tie_margin,
        "exactness": dict(exactness),
        "invalid_prompt_occurrences": invalid_prompts,
        "grouped": grouped,
        "plots": make_plot(grouped, args.out_dir),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Aggregated {payload['num_complete_runs']} runs; missing {len(missing)}")
    print(args.out_dir / "summary.json")


if __name__ == "__main__":
    main()
