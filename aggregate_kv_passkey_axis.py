#!/usr/bin/env python3
"""Compare passkey K/V precision preferences across key quantization axes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from spec_kv_statistics import bootstrap_mean_ci


EVALUATOR_VERSION = "kv_multiple_choice_cached_v2"
DEFAULT_GENERATOR_VERSION = "synthetic_associative_passkey_v3"
CONFIGS = ("none", "k4v4", "k4v2", "k2v4", "k2v2")


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


def parse_int_list(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def load_axis(
    root: Path,
    *,
    axis: str,
    context: int,
    seeds: Sequence[int],
    examples_per_run: int,
    generator_version: str,
    passkey_variant: str,
    passkey_score: str,
    num_choices: int,
) -> Tuple[List[Dict[str, str]], Dict[str, List[float]], List[str]]:
    rows: List[Dict[str, str]] = []
    savings: Dict[str, List[float]] = {config: [] for config in CONFIGS}
    missing: List[str] = []
    for seed in seeds:
        seed_dir = root / f"ctx_{context}" / f"seed_{seed}"
        summary_path = seed_dir / "summary.json"
        rows_path = seed_dir / "example_rows.csv"
        if not summary_path.exists() or not rows_path.exists():
            missing.append(str(seed_dir))
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        runtime = summary.get("runtime", {})
        config = summary.get("config", {})
        checks = {
            "evaluator": (runtime.get("evaluator_version"), EVALUATOR_VERSION),
            "generator": (runtime.get("task_generator_version"), generator_version),
            "task": (summary.get("task"), "passkey"),
            "primary_metric": (summary.get("primary_metric"), f"{passkey_score}_accuracy"),
            "axis": (runtime.get("key_quant_axis"), axis),
            "context": (int(config.get("max_prompt_tokens", -1)), context),
            "examples": (int(summary.get("num_examples", -1)), examples_per_run),
            "choices": (int(config.get("passkey_num_choices", -1)), num_choices),
            "variant": (config.get("passkey_variant"), passkey_variant),
            "score": (config.get("passkey_score"), passkey_score),
        }
        failed = {name: values for name, values in checks.items() if values[0] != values[1]}
        if failed:
            raise ValueError(f"Integrity check failed for {summary_path}: {failed}")
        run_rows = read_csv(rows_path)
        expected_rows = examples_per_run * len(CONFIGS)
        if len(run_rows) != expected_rows:
            raise ValueError(
                f"Expected {expected_rows} rows in {rows_path}, observed {len(run_rows)}"
            )
        for row in run_rows:
            if row.get("config") not in CONFIGS:
                raise ValueError(f"Unexpected config in {rows_path}: {row.get('config')}")
            tagged = dict(row)
            tagged["axis"] = axis
            rows.append(tagged)
        for name in CONFIGS:
            if name not in summary.get("summaries", {}):
                raise ValueError(f"Missing {name} summary in {summary_path}")
            savings[name].append(float(summary["summaries"][name]["cache_saved_fraction"]))
    return rows, savings, missing


def paired_values(
    rows: Iterable[Mapping[str, str]],
    *,
    score_field: str,
) -> Dict[Tuple[str, str], Dict[Tuple[str, str], float]]:
    paired: Dict[Tuple[str, str], Dict[Tuple[str, str], float]] = {}
    for row in rows:
        key = (str(row["seed"]), str(row["source_idx"]))
        paired.setdefault(key, {})[(str(row["axis"]), str(row["config"]))] = float(
            row[score_field]
        )
    return paired


def make_plot(axis_results: Sequence[Mapping[str, Any]], out_dir: Path) -> List[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    axes = ("per_channel", "per_token")
    configs = ("k4v2", "k2v4")
    colors = {"k4v2": "#244062", "k2v4": "#d1495b"}
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    width = 0.34
    for offset, config in enumerate(configs):
        subset = {
            str(row["axis"]): row
            for row in axis_results
            if str(row["config"]) == config
        }
        positions = [index + (offset - 0.5) * width for index in range(len(axes))]
        values = [100.0 * float(subset[name]["accuracy_mean"]) for name in axes]
        errors = [
            [100.0 * (float(subset[name]["accuracy_mean"]) - float(subset[name]["accuracy_ci_low"])) for name in axes],
            [100.0 * (float(subset[name]["accuracy_ci_high"]) - float(subset[name]["accuracy_mean"])) for name in axes],
        ]
        axis.bar(
            positions,
            values,
            width=width,
            yerr=errors,
            capsize=4,
            color=colors[config],
            label=config.upper(),
        )
    axis.set_xticks(range(len(axes)), ["Per-channel keys", "Per-token keys"])
    axis.set_ylabel("Normalized retrieval accuracy (%)")
    axis.set_ylim(0.0, 103.0)
    axis.grid(axis="y", alpha=0.22)
    axis.legend(frameon=False)
    axis.set_title("Key Geometry Changes the K/V Precision Tradeoff", fontweight="bold")
    figure.tight_layout()
    paths = []
    for extension in ("png", "pdf"):
        path = out_dir / f"passkey_axis_interaction.{extension}"
        figure.savefig(path, dpi=240, bbox_inches="tight")
        paths.append(str(path))
    plt.close(figure)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per_channel_root", type=Path, required=True)
    parser.add_argument("--per_token_root", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--context", type=int, default=16384)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--expected_examples_per_run", type=int, default=16)
    parser.add_argument("--expected_num_choices", type=int, default=16)
    parser.add_argument("--expected_generator_version", default=DEFAULT_GENERATOR_VERSION)
    parser.add_argument("--expected_passkey_variant", default="confusable_records")
    parser.add_argument("--expected_passkey_score", default="normalized")
    parser.add_argument("--bootstrap_samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--require_complete", action="store_true")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    seeds = parse_int_list(args.seeds)
    all_rows: List[Dict[str, str]] = []
    all_savings: Dict[str, Dict[str, List[float]]] = {}
    missing: List[str] = []
    for axis, root in (
        ("per_channel", args.per_channel_root),
        ("per_token", args.per_token_root),
    ):
        rows, savings, axis_missing = load_axis(
            root,
            axis=axis,
            context=args.context,
            seeds=seeds,
            examples_per_run=args.expected_examples_per_run,
            generator_version=args.expected_generator_version,
            passkey_variant=args.expected_passkey_variant,
            passkey_score=args.expected_passkey_score,
            num_choices=args.expected_num_choices,
        )
        all_rows.extend(rows)
        all_savings[axis] = savings
        missing.extend(axis_missing)
    if args.require_complete and missing:
        raise ValueError(f"Axis sweep is incomplete: {missing}")
    score_field = f"{args.expected_passkey_score}_correct"
    paired = paired_values(all_rows, score_field=score_field)
    required = {
        (axis, config)
        for axis in ("per_channel", "per_token")
        for config in CONFIGS
    }
    complete = {key: values for key, values in paired.items() if required <= set(values)}
    expected_pairs = len(seeds) * args.expected_examples_per_run
    if args.require_complete and len(complete) != expected_pairs:
        raise ValueError(f"Expected {expected_pairs} paired examples, observed {len(complete)}")

    axis_results: List[Dict[str, Any]] = []
    contrasts: List[Dict[str, Any]] = []
    for axis_index, axis in enumerate(("per_channel", "per_token")):
        for config_index, config in enumerate(CONFIGS):
            values = [items[(axis, config)] for items in complete.values()]
            stats = bootstrap_mean_ci(
                values,
                seed=args.seed + axis_index * 20 + config_index,
                samples=args.bootstrap_samples,
            )
            axis_results.append(
                {
                    "axis": axis,
                    "config": config,
                    "paired_count": len(values),
                    "accuracy_mean": stats["mean"],
                    "accuracy_ci_low": stats["ci_low"],
                    "accuracy_ci_high": stats["ci_high"],
                    "cache_saved_fraction_mean": sum(all_savings[axis][config])
                    / max(1, len(all_savings[axis][config])),
                }
            )
        differences = [
            items[(axis, "k4v2")] - items[(axis, "k2v4")]
            for items in complete.values()
        ]
        stats = bootstrap_mean_ci(
            differences,
            seed=args.seed + 100 + axis_index,
            samples=args.bootstrap_samples,
        )
        contrasts.append(
            {
                "contrast": "k4v2_minus_k2v4",
                "axis": axis,
                "paired_count": len(differences),
                **stats,
            }
        )
    interactions = [
        (items[("per_channel", "k4v2")] - items[("per_channel", "k2v4")])
        - (items[("per_token", "k4v2")] - items[("per_token", "k2v4")])
        for items in complete.values()
    ]
    interaction_stats = bootstrap_mean_ci(
        interactions,
        seed=args.seed + 200,
        samples=args.bootstrap_samples,
    )
    contrasts.append(
        {
            "contrast": "axis_by_allocation_interaction",
            "axis": "per_channel_minus_per_token",
            "paired_count": len(interactions),
            **interaction_stats,
        }
    )
    paired_rows = []
    for (seed, source_idx), items in sorted(complete.items(), key=lambda item: (int(item[0][0]), int(item[0][1]))):
        paired_rows.append(
            {
                "seed": seed,
                "source_idx": source_idx,
                "per_channel_k4v2": items[("per_channel", "k4v2")],
                "per_channel_k2v4": items[("per_channel", "k2v4")],
                "per_token_k4v2": items[("per_token", "k4v2")],
                "per_token_k2v4": items[("per_token", "k2v4")],
                "interaction": (
                    items[("per_channel", "k4v2")] - items[("per_channel", "k2v4")]
                    - items[("per_token", "k4v2")]
                    + items[("per_token", "k2v4")]
                ),
            }
        )
    write_csv(args.out_dir / "axis_results.csv", axis_results)
    write_csv(args.out_dir / "paired_examples.csv", paired_rows)
    plots = make_plot(axis_results, args.out_dir)
    summary = {
        "evaluator_version": EVALUATOR_VERSION,
        "task_generator_version": args.expected_generator_version,
        "context": args.context,
        "expected_seeds": seeds,
        "expected_examples_per_run": args.expected_examples_per_run,
        "paired_count": len(complete),
        "missing_runs": missing,
        "complete_run_gate": not missing and len(complete) == expected_pairs,
        "axis_results": axis_results,
        "contrasts": contrasts,
        "plots": plots,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
