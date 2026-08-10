#!/usr/bin/env python3
"""Aggregate joint target/draft KV quantization under speculative decoding."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from aggregate_value_precision_sweep import exactness_status, read_csv, read_json, write_csv
from spec_kv_statistics import bootstrap_acceptance_contrast


EVALUATOR_VERSION = "cached_dynamic_v5_joint_target_draft"
BASELINE_CONFIG = "target_none__draft_none"


def collect_prompt_effects(
    rows: Iterable[Dict[str, str]],
    *,
    config_names: Iterable[str],
    baseline_name: str,
    tie_margin: float,
) -> Tuple[
    Dict[str, List[Tuple[Dict[str, str], Dict[str, str]]]],
    Dict[str, Counter[str]],
    int,
]:
    expected = set(config_names)
    by_prompt: Dict[str, Dict[str, Dict[str, str]]] = defaultdict(dict)
    exactness: Dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        name = row.get("config", "")
        if name not in expected:
            continue
        by_prompt[row["prompt_idx"]][name] = row
        exactness[name][exactness_status(row, tie_margin=tie_margin)] += 1

    effects: Dict[str, List[Tuple[Dict[str, str], Dict[str, str]]]] = defaultdict(list)
    invalid_baseline_prompts = 0
    for values in by_prompt.values():
        baseline = values.get(baseline_name)
        if baseline is None:
            continue
        if exactness_status(baseline, tie_margin=tie_margin) == "non_tie_or_unknown":
            invalid_baseline_prompts += 1
            continue
        for name in expected - {baseline_name}:
            if name in values:
                effects[name].append((values[name], baseline))
    return effects, exactness, invalid_baseline_prompts


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
        subset = [
            row
            for row in rows
            if int(row["context"]) == context and row["config"] != BASELINE_CONFIG
        ]
        scatter = axis.scatter(
            [100.0 * float(row["total_cache_saved_fraction"]) for row in subset],
            [100.0 * float(row["paired_acceptance_delta_mean"]) for row in subset],
            c=[float(row["bf16_reference_token_match_mean"]) for row in subset],
            cmap="viridis",
            vmin=0.8,
            vmax=1.0,
            s=62,
            edgecolors="#222222",
            linewidths=0.5,
        )
        for row in subset:
            axis.annotate(
                f"T:{row['target_config']}\nD:{row['draft_config']}",
                (
                    100.0 * float(row["total_cache_saved_fraction"]),
                    100.0 * float(row["paired_acceptance_delta_mean"]),
                ),
                fontsize=7,
            )
        axis.axhline(0.0, color="#222222", linewidth=1)
        axis.set_title(f"Context {context:,}")
        axis.set_xlabel("Total target + draft KV saved (%)")
        axis.set_ylabel("Acceptance change vs BF16 (pp)")
        axis.grid(alpha=0.22)
        if subset:
            fig.colorbar(scatter, ax=axis, label="Token match to BF16 target")
    fig.suptitle("Joint Target/Draft KV Quantization", fontweight="bold")
    fig.tight_layout()
    paths = []
    for extension in ("png", "pdf"):
        path = out_dir / f"joint_target_draft_pareto.{extension}"
        fig.savefig(path, dpi=240, bbox_inches="tight")
        paths.append(str(path))
    plt.close(fig)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep_dir", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--exactness_tie_margin", type=float, default=1e-3)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    run_rows: List[Dict[str, Any]] = []
    prompt_effects: Dict[
        Tuple[int, str], List[Tuple[Dict[str, str], Dict[str, str]]]
    ] = defaultdict(list)
    fidelity_rows: Dict[Tuple[int, str], List[Dict[str, str]]] = defaultdict(list)
    exactness_by_config: Dict[str, Counter[str]] = defaultdict(Counter)
    invalid_baseline_prompts = 0
    missing = []

    for seed_dir in sorted(args.sweep_dir.glob("ctx_*/seed_*")):
        summary_path = seed_dir / "summary.json"
        benchmark_path = seed_dir / "benchmark_rows.csv"
        if not summary_path.exists() or not benchmark_path.exists():
            missing.append(str(seed_dir))
            continue
        summary = read_json(summary_path)
        version = summary.get("runtime", {}).get("evaluator_version")
        if version != EVALUATOR_VERSION:
            raise ValueError(f"Unexpected evaluator {version!r} in {summary_path}")
        config = summary["config"]
        context = int(config["prompt_len"])
        seed = int(config["seed"])
        names = list(summary["quant_configs"])
        if BASELINE_CONFIG not in names:
            raise ValueError(f"Missing joint BF16 baseline in {summary_path}")
        raw_rows = read_csv(benchmark_path)
        effects, exactness, invalid = collect_prompt_effects(
            raw_rows,
            config_names=names,
            baseline_name=BASELINE_CONFIG,
            tie_margin=args.exactness_tie_margin,
        )
        invalid_baseline_prompts += invalid
        for name, counts in exactness.items():
            exactness_by_config[name].update(counts)
        for name, values in effects.items():
            prompt_effects[(context, name)].extend(values)
        for raw in raw_rows:
            if raw.get("config") in names:
                fidelity_rows[(context, raw["config"])].append(raw)

        for name in names:
            metrics = summary["summaries"][name]
            run_rows.append(
                {
                    "context": context,
                    "seed": seed,
                    "dataset_name": config["dataset_name"],
                    "config": name,
                    "target_config": metrics["target_config"],
                    "draft_config": metrics["draft_config"],
                    "target_k_bits_mean": metrics["target_allocation/k_bits_mean"],
                    "target_v_bits_mean": metrics["target_allocation/v_bits_mean"],
                    "draft_k_bits_mean": metrics["allocation/k_bits_mean"],
                    "draft_v_bits_mean": metrics["allocation/v_bits_mean"],
                    "accept_rate": metrics["overall_accept_rate"],
                    "round_js": metrics["round_js"],
                    "target_cache_saved_fraction": metrics[
                        "target_cache_saved_fraction"
                    ],
                    "draft_cache_saved_fraction": metrics[
                        "draft_cache_saved_fraction"
                    ],
                    "total_cache_saved_fraction": metrics[
                        "total_cache_saved_fraction"
                    ],
                }
            )

    by_config: Dict[Tuple[int, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        by_config[(int(row["context"]), str(row["config"]))].append(row)

    grouped: List[Dict[str, Any]] = []
    for (context, name), values in sorted(by_config.items()):
        if name == BASELINE_CONFIG:
            acceptance = {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
        else:
            acceptance = bootstrap_acceptance_contrast(
                prompt_effects[(context, name)],
                (1.0, -1.0),
                seed=context * 100 + sum(map(ord, name)),
            )
        raw = fidelity_rows[(context, name)]
        sequence_matches = [float(row["matches_target_greedy"]) for row in raw]
        token_matches = [
            float(row.get("target_token_match_fraction", row["matches_target_greedy"]))
            for row in raw
        ]
        counts = exactness_by_config[name]
        grouped.append(
            {
                "context": context,
                "config": name,
                "target_config": values[0]["target_config"],
                "draft_config": values[0]["draft_config"],
                "num_seeds": len(values),
                "paired_prompt_count": len(prompt_effects[(context, name)]),
                "paired_acceptance_delta_mean": acceptance["mean"],
                "paired_acceptance_delta_ci_low": acceptance["ci_low"],
                "paired_acceptance_delta_ci_high": acceptance["ci_high"],
                "accept_rate_mean": statistics.mean(
                    float(row["accept_rate"]) for row in values
                ),
                "round_js_mean": statistics.mean(
                    float(row["round_js"]) for row in values
                ),
                "bf16_reference_sequence_match_mean": statistics.mean(
                    sequence_matches
                ),
                "bf16_reference_token_match_mean": statistics.mean(token_matches),
                "bf16_reference_exact_count": counts["exact"],
                "bf16_reference_tie_count": counts["numerical_tie"],
                "bf16_reference_non_tie_count": counts["non_tie_or_unknown"],
                "target_cache_saved_fraction": statistics.mean(
                    float(row["target_cache_saved_fraction"]) for row in values
                ),
                "draft_cache_saved_fraction": statistics.mean(
                    float(row["draft_cache_saved_fraction"]) for row in values
                ),
                "total_cache_saved_fraction": statistics.mean(
                    float(row["total_cache_saved_fraction"]) for row in values
                ),
            }
        )

    if not grouped:
        raise ValueError("No complete joint target/draft outputs were found.")
    write_csv(args.out_dir / "run_results.csv", run_rows)
    write_csv(args.out_dir / "grouped_results.csv", grouped)
    payload = {
        "evaluator_version": EVALUATOR_VERSION,
        "baseline_config": BASELINE_CONFIG,
        "num_complete_runs": len(
            {(row["context"], row["seed"]) for row in run_rows}
        ),
        "missing_runs": missing,
        "exactness_tie_margin": args.exactness_tie_margin,
        "invalid_bf16_baseline_prompt_occurrences": invalid_baseline_prompts,
        "exactness_by_config": {
            name: dict(counts) for name, counts in exactness_by_config.items()
        },
        "grouped": grouped,
        "plots": make_plot(grouped, args.out_dir),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Aggregated {payload['num_complete_runs']} joint runs")
    print(args.out_dir / "summary.json")


if __name__ == "__main__":
    main()
