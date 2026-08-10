#!/usr/bin/env python3
"""Compare ordinary-LM KV quantization sensitivity by speculative model role."""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from aggregate_kivi_cross_family_meta import DEFAULT_PAIRS


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    values = list(rows)
    if not values:
        return
    fields: List[str] = []
    for row in values:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(values)


def bootstrap_macro_ci(
    values: Sequence[float], *, seed: int, samples: int = 10_000
) -> Dict[str, float]:
    if not values:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    if len(values) == 1:
        value = float(values[0])
        return {"mean": value, "ci_low": value, "ci_high": value}
    rng = random.Random(seed)
    estimates = sorted(
        statistics.mean(float(values[rng.randrange(len(values))]) for _ in values)
        for _ in range(samples)
    )
    return {
        "mean": statistics.mean(map(float, values)),
        "ci_low": estimates[int(0.025 * samples)],
        "ci_high": estimates[min(samples - 1, int(0.975 * samples))],
    }


def collect_role_rows(
    root: Path, expected_pairs: Sequence[str]
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    missing = []
    run_counts: Dict[str, Dict[str, int]] = {}
    for pair in expected_pairs:
        draft_path = root / "quality" / pair / "aggregate" / "summary.json"
        target_path = root / "target_quality" / pair / "aggregate" / "summary.json"
        absent = [str(path) for path in (draft_path, target_path) if not path.exists()]
        if absent:
            missing.extend(absent)
            continue
        draft = read_json(draft_path)
        target = read_json(target_path)
        run_counts[pair] = {
            "draft": int(draft.get("num_complete_runs", 0)),
            "target": int(target.get("num_complete_runs", 0)),
        }
        draft_rows = {
            (int(row["context"]), str(row["config"])): row
            for row in draft.get("grouped", [])
        }
        target_rows = {
            (int(row["context"]), str(row["config"])): row
            for row in target.get("grouped", [])
        }
        for context, config in sorted(draft_rows.keys() & target_rows.keys()):
            draft_row = draft_rows[(context, config)]
            target_row = target_rows[(context, config)]
            target_kl = float(target_row["kl_p_to_q_mean"])
            draft_kl = float(draft_row["kl_p_to_q_mean"])
            rows.append(
                {
                    "pair": pair,
                    "context": context,
                    "config": config,
                    "k_bits": draft_row["k_bits"],
                    "v_bits": draft_row["v_bits"],
                    "cache_saved_fraction": target_row["cache_saved_fraction"],
                    "target_kl_mean": target_kl,
                    "target_kl_ci_low": target_row["kl_p_to_q_ci_low"],
                    "target_kl_ci_high": target_row["kl_p_to_q_ci_high"],
                    "draft_kl_mean": draft_kl,
                    "draft_kl_ci_low": draft_row["kl_p_to_q_ci_low"],
                    "draft_kl_ci_high": draft_row["kl_p_to_q_ci_high"],
                    "target_minus_draft_kl": target_kl - draft_kl,
                    "target_top1_match": target_row["top1_match_mean"],
                    "draft_top1_match": draft_row["top1_match_mean"],
                    "target_delta_nll": target_row["delta_nll_mean"],
                    "draft_delta_nll": draft_row["delta_nll_mean"],
                }
            )
    audit = {
        "expected_pairs": list(expected_pairs),
        "complete_pairs": sorted({str(row["pair"]) for row in rows}),
        "missing_artifacts": missing,
        "run_counts": run_counts,
    }
    return rows, audit


def aggregate_role_differences(rows: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple[int, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["context"]), str(row["config"]))].append(row)

    output: List[Dict[str, Any]] = []
    for (context, config), values in sorted(grouped.items()):
        target = bootstrap_macro_ci(
            [float(row["target_kl_mean"]) for row in values],
            seed=context + sum(map(ord, config)),
        )
        draft = bootstrap_macro_ci(
            [float(row["draft_kl_mean"]) for row in values],
            seed=2 * context + sum(map(ord, config)),
        )
        difference = bootstrap_macro_ci(
            [float(row["target_minus_draft_kl"]) for row in values],
            seed=3 * context + sum(map(ord, config)),
        )
        output.append(
            {
                "context": context,
                "config": config,
                "num_pairs": len(values),
                "pairs": ";".join(sorted(str(row["pair"]) for row in values)),
                "cache_saved_fraction_macro_mean": statistics.mean(
                    float(row["cache_saved_fraction"]) for row in values
                ),
                "target_kl_macro_mean": target["mean"],
                "target_kl_macro_ci_low": target["ci_low"],
                "target_kl_macro_ci_high": target["ci_high"],
                "draft_kl_macro_mean": draft["mean"],
                "draft_kl_macro_ci_low": draft["ci_low"],
                "draft_kl_macro_ci_high": draft["ci_high"],
                "target_minus_draft_kl_macro_mean": difference["mean"],
                "target_minus_draft_kl_macro_ci_low": difference["ci_low"],
                "target_minus_draft_kl_macro_ci_high": difference["ci_high"],
                "num_pairs_target_more_sensitive": sum(
                    float(row["target_minus_draft_kl"]) > 0.0 for row in values
                ),
                "target_top1_match_macro_mean": statistics.mean(
                    float(row["target_top1_match"]) for row in values
                ),
                "draft_top1_match_macro_mean": statistics.mean(
                    float(row["draft_top1_match"]) for row in values
                ),
            }
        )
    return output


def make_plot(rows: List[Dict[str, Any]], out_dir: Path) -> List[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    contexts = sorted({int(row["context"]) for row in rows})
    fig, axes = plt.subplots(
        1, len(contexts), figsize=(5.4 * len(contexts), 4.8), squeeze=False
    )
    label_offsets = {
        "k2v2": (-43, 6),
        "k2v4": (5, 4),
        "k3v4": (5, -14),
        "k4v2": (5, -12),
        "k4v3": (5, 4),
        "k4v4": (5, 8),
        "k4v8": (5, -13),
        "k8v4": (5, -13),
    }
    for axis, context in zip(axes[0], contexts):
        subset = [row for row in rows if int(row["context"]) == context]
        values = [
            float(row[field])
            for row in subset
            for field in ("target_kl_macro_mean", "draft_kl_macro_mean")
            if float(row[field]) > 0.0
        ]
        minimum = min(values) / 1.6
        maximum = max(
            values + [1e-4]
        ) * 1.35
        axis.plot(
            [minimum, maximum],
            [minimum, maximum],
            "--",
            color="#777777",
            linewidth=1.2,
            label="Equal sensitivity",
        )
        for row in subset:
            config = str(row["config"])
            axis.scatter(
                float(row["draft_kl_macro_mean"]),
                float(row["target_kl_macro_mean"]),
                color="#D1495B",
                edgecolors="#222222",
                linewidths=0.4,
                s=55,
            )
            axis.annotate(
                config.upper(),
                (
                    float(row["draft_kl_macro_mean"]),
                    float(row["target_kl_macro_mean"]),
                ),
                fontsize=8,
                xytext=label_offsets.get(config, (5, 4)),
                textcoords="offset points",
            )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlim(minimum, maximum)
        axis.set_ylim(minimum, maximum)
        axis.set_title(f"Context {context:,}")
        axis.set_xlabel("Smaller draft model KL")
        axis.set_ylabel("Larger target model KL")
        axis.grid(alpha=0.22, which="both")
        axis.legend(loc="upper left", frameon=False, fontsize=8)
    fig.suptitle(
        "Larger Target Models Are More Robust to the Same KV Quantization",
        fontweight="bold",
    )
    fig.tight_layout()
    paths = []
    for extension in ("png", "pdf"):
        path = out_dir / f"target_vs_draft_quality.{extension}"
        fig.savefig(path, dpi=240, bbox_inches="tight")
        paths.append(str(path))
    plt.close(fig)
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("outputs/kivi_objective_cross_family")
    )
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--expected_pairs", default=",".join(DEFAULT_PAIRS))
    parser.add_argument("--allow_incomplete", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    expected = [value.strip() for value in args.expected_pairs.split(",") if value.strip()]
    rows, audit = collect_role_rows(args.root, expected)
    if audit["missing_artifacts"] and not args.allow_incomplete:
        raise ValueError(
            "Missing target/draft quality artifacts:\n"
            + "\n".join(audit["missing_artifacts"])
        )
    if not rows:
        raise ValueError("No matched target/draft ordinary-quality rows were found.")
    grouped = aggregate_role_differences(rows)
    write_csv(args.out_dir / "pair_role_quality.csv", rows)
    write_csv(args.out_dir / "role_quality_macro.csv", grouped)
    payload = {
        "audit": audit,
        "grouped": grouped,
        "plots": make_plot(grouped, args.out_dir),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Compared target and draft quality across {len(audit['complete_pairs'])} pairs")
    print(args.out_dir / "summary.json")


if __name__ == "__main__":
    main()
