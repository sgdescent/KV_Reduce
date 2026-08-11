#!/usr/bin/env python3
"""Strictly aggregate prompt-paired KV quantizer geometry sweeps."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from spec_kv_statistics import bootstrap_mean_ci


EVALUATOR_VERSION = "cached_dynamic_v6_sequential_target"


def parse_int_list(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


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


def paired_values(
    rows: Iterable[Dict[str, Any]],
    *,
    treatment: int,
    reference: int,
    config: str,
    metric: str,
) -> List[float]:
    grouped: Dict[Tuple[int, int], Dict[int, float]] = defaultdict(dict)
    for row in rows:
        if str(row["config"]) != config:
            continue
        grouped[(int(row["seed"]), int(row["prompt_idx"]))][
            int(row["treatment"])
        ] = float(row[metric])
    return [
        values[treatment] - values[reference]
        for values in grouped.values()
        if treatment in values and reference in values
    ]


def allocation_contrast(
    rows: Iterable[Dict[str, Any]],
    *,
    treatment: int,
    config_a: str,
    config_b: str,
) -> List[float]:
    grouped: Dict[Tuple[int, int], Dict[str, float]] = defaultdict(dict)
    for row in rows:
        if int(row["treatment"]) != treatment:
            continue
        grouped[(int(row["seed"]), int(row["prompt_idx"]))][str(row["config"])] = float(
            row["accept_rate"]
        )
    return [
        values[config_a] - values[config_b]
        for values in grouped.values()
        if config_a in values and config_b in values
    ]


def collect(
    *,
    root: Path,
    treatment_prefix: str,
    treatment_config_key: str,
    treatments: Sequence[int],
    seeds: Sequence[int],
    expected_prompts: int,
) -> Tuple[List[Dict[str, Any]], Dict[Tuple[int, int], Dict[str, Any]], List[str]]:
    rows: List[Dict[str, Any]] = []
    summaries: Dict[Tuple[int, int], Dict[str, Any]] = {}
    sources: List[str] = []
    offsets_by_seed: Dict[int, set[int]] = defaultdict(set)
    for treatment in treatments:
        for seed in seeds:
            run_dir = root / f"{treatment_prefix}_{treatment}" / f"seed_{seed}"
            summary_path = run_dir / "summary.json"
            rows_path = run_dir / "benchmark_rows.csv"
            if not summary_path.exists() or not rows_path.exists():
                raise ValueError(f"Missing paired geometry run: {run_dir}")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            runtime = summary.get("runtime", {})
            config = summary.get("config", {})
            if runtime.get("evaluator_version") != EVALUATOR_VERSION:
                raise ValueError(f"Unexpected evaluator in {summary_path}: {runtime}")
            if runtime.get("target_verification_mode") != "sequential":
                raise ValueError(f"Non-sequential verifier in {summary_path}")
            if summary.get("target_quant_configs") != ["none"]:
                raise ValueError(f"Target cache is not BF16-only in {summary_path}")
            if int(summary.get("num_prompts", -1)) != expected_prompts:
                raise ValueError(f"Underfilled run in {summary_path}")
            if int(config.get("seed", -1)) != seed:
                raise ValueError(f"Seed mismatch in {summary_path}")
            if int(runtime.get(treatment_config_key, -1)) != treatment:
                raise ValueError(f"Treatment mismatch in {summary_path}")
            offsets_by_seed[seed].add(int(config["skip_prompts"]))
            raw_rows = read_csv(rows_path)
            expected_rows = expected_prompts * len(summary["draft_quant_configs"])
            if len(raw_rows) != expected_rows:
                raise ValueError(
                    f"Row-count mismatch in {rows_path}: {len(raw_rows)} != {expected_rows}"
                )
            for raw in raw_rows:
                if float(raw["matches_target_greedy"]) != 1.0:
                    raise ValueError(f"Target mismatch in {rows_path}")
                if int(float(raw["first_target_mismatch"])) != -1:
                    raise ValueError(f"Target mismatch marker in {rows_path}")
                rows.append({"treatment": treatment, "seed": seed, **raw})
            summaries[(treatment, seed)] = summary
            sources.extend((str(summary_path), str(rows_path)))
    unpaired_offsets = {
        seed: sorted(offsets) for seed, offsets in offsets_by_seed.items() if len(offsets) != 1
    }
    if unpaired_offsets:
        raise ValueError(f"Treatments do not share prompts within seed: {unpaired_offsets}")
    return rows, summaries, sources


def make_plot(rows: Sequence[Dict[str, Any]], out_dir: Path, label: str) -> List[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    configs = ["k8v4", "k4v8", "k4v4", "k2v4", "k4v2"]
    fig, axis = plt.subplots(figsize=(7.6, 4.8))
    for config in configs:
        subset = [row for row in rows if row["config"] == config]
        if not subset:
            continue
        axis.errorbar(
            [row["treatment"] for row in subset],
            [100.0 * row["accept_rate_mean"] for row in subset],
            yerr=[
                [100.0 * (row["accept_rate_mean"] - row["accept_rate_ci_low"]) for row in subset],
                [100.0 * (row["accept_rate_ci_high"] - row["accept_rate_mean"]) for row in subset],
            ],
            marker="o",
            capsize=3,
            label=config.upper(),
        )
    axis.set_xlabel(label)
    axis.set_ylabel("Speculative acceptance (%)")
    axis.grid(alpha=0.22)
    axis.legend(ncol=3, frameon=False)
    axis.set_title(f"Prompt-Paired {label} Sweep", fontweight="bold")
    fig.tight_layout()
    paths = []
    for extension in ("png", "pdf"):
        path = out_dir / f"paired_{label.lower().replace(' ', '_')}.{extension}"
        fig.savefig(path, dpi=240, bbox_inches="tight")
        paths.append(str(path))
    plt.close(fig)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--treatment_prefix", required=True)
    parser.add_argument("--treatment_config_key", required=True)
    parser.add_argument("--treatment_label", required=True)
    parser.add_argument("--treatments", required=True)
    parser.add_argument("--reference", type=int, required=True)
    parser.add_argument("--expected_seeds", default="0,1,2")
    parser.add_argument("--expected_prompts", type=int, default=32)
    parser.add_argument("--bootstrap_samples", type=int, default=30_000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    treatments = parse_int_list(args.treatments)
    seeds = parse_int_list(args.expected_seeds)
    if args.reference not in treatments:
        raise ValueError("Reference treatment must be included in treatments.")
    raw_rows, summaries, sources = collect(
        root=args.root,
        treatment_prefix=args.treatment_prefix,
        treatment_config_key=args.treatment_config_key,
        treatments=treatments,
        seeds=seeds,
        expected_prompts=args.expected_prompts,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    grouped = []
    configs = sorted({str(row["config"]) for row in raw_rows})
    for treatment in treatments:
        for config in configs:
            values = [
                float(row["accept_rate"])
                for row in raw_rows
                if int(row["treatment"]) == treatment and row["config"] == config
            ]
            estimate = bootstrap_mean_ci(
                values,
                seed=args.seed + len(grouped),
                samples=args.bootstrap_samples,
            )
            delta_values = paired_values(
                raw_rows,
                treatment=treatment,
                reference=args.reference,
                config=config,
                metric="accept_rate",
            )
            delta = bootstrap_mean_ci(
                delta_values,
                seed=args.seed + 1000 + len(grouped),
                samples=args.bootstrap_samples,
            )
            run_summaries = [summaries[(treatment, seed)]["summaries"][config] for seed in seeds]
            grouped.append(
                {
                    "treatment": treatment,
                    "config": config,
                    "paired_prompts": len(delta_values),
                    "accept_rate_mean": estimate["mean"],
                    "accept_rate_ci_low": estimate["ci_low"],
                    "accept_rate_ci_high": estimate["ci_high"],
                    "delta_vs_reference_mean": delta["mean"],
                    "delta_vs_reference_ci_low": delta["ci_low"],
                    "delta_vs_reference_ci_high": delta["ci_high"],
                    "draft_cache_saved_fraction": sum(
                        float(summary["draft_cache_saved_fraction"])
                        for summary in run_summaries
                    )
                    / len(run_summaries),
                    "total_cache_saved_fraction": sum(
                        float(summary["total_cache_saved_fraction"])
                        for summary in run_summaries
                    )
                    / len(run_summaries),
                }
            )
    contrasts = []
    for treatment in treatments:
        for config_a, config_b in (("k8v4", "k4v8"), ("k4v2", "k2v4")):
            if config_a not in configs or config_b not in configs:
                continue
            values = allocation_contrast(
                raw_rows,
                treatment=treatment,
                config_a=config_a,
                config_b=config_b,
            )
            estimate = bootstrap_mean_ci(
                values,
                seed=args.seed + 2000 + len(contrasts),
                samples=args.bootstrap_samples,
            )
            contrasts.append(
                {
                    "treatment": treatment,
                    "config_a": config_a,
                    "config_b": config_b,
                    "paired_prompts": len(values),
                    "acceptance_a_minus_b_mean": estimate["mean"],
                    "acceptance_a_minus_b_ci_low": estimate["ci_low"],
                    "acceptance_a_minus_b_ci_high": estimate["ci_high"],
                }
            )
    write_csv(args.out_dir / "grouped_results.csv", grouped)
    write_csv(args.out_dir / "allocation_contrasts.csv", contrasts)
    payload = {
        "runtime": {
            "source_evaluator_version": EVALUATOR_VERSION,
            "paired_prompt_gate": True,
            "exact_target_gate": True,
            "complete_run_gate": True,
        },
        "treatment_prefix": args.treatment_prefix,
        "treatment_config_key": args.treatment_config_key,
        "treatment_label": args.treatment_label,
        "treatments": treatments,
        "reference": args.reference,
        "expected_seeds": seeds,
        "expected_prompts": args.expected_prompts,
        "sources": sources,
        "grouped": grouped,
        "allocation_contrasts": contrasts,
        "plots": make_plot(grouped, args.out_dir, args.treatment_label),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(args.out_dir / "summary.json")


if __name__ == "__main__":
    main()
