#!/usr/bin/env python3
"""Aggregate validated objective-aware KV matrices into paper artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List


MATRIX_LABELS = {
    "qwen25_objective_matrix_v1": "Qwen / WikiText / top-8 / raw",
    "qwen25_objective_ucb_matrix_v1": "Qwen / WikiText / top-8 / rate-UCB",
    "qwen25_objective_c4_matrix_v1": "Qwen / C4 / top-8 / raw",
    "qwen25_objective_ucb_profile64_nested_matrix_v1": "Qwen / WikiText / top-8 / 64-cal",
    "qwen25_accept_mass_matrix_v1": "Qwen / WikiText / top-8 / mass-UCB",
    "qwen25_all_layers_mass_matrix_v1": "Qwen / WikiText / all-layer / mass-UCB",
    "qwen25_all_layers_eager_1k_v1": "Qwen / WikiText / all-layer / eager backend",
    "llama_mass_matrix_v1": "Llama / WikiText / top-8 / mass-UCB",
    "olmo2_mass_matrix_v1": "OLMo / WikiText / top-8 / mass-UCB",
    "qwen25_all_layers_long_context_v1": "Qwen / WikiText / all-layer / 8K-16K",
    "qwen25_all_layers_32k_v1": "Qwen / WikiText / all-layer / 32K",
    "qwen25_all_layers_pg19_long_v1": "Qwen / PG19 / all-layer / 16K-32K",
    "qwen25_all_layers_gsm8k_v1": "Qwen / GSM8K answers / all-layer",
    "qwen25_all_layers_humaneval_v1": "Qwen / HumanEval prompts / all-layer",
    "qwen25_all_layers_profile64_mass_matrix_v1": "Qwen / WikiText / all-layer / 64-cal",
}

FINAL_LABELS = {
    "qwen25_objective_1k_v1": "Qwen / WikiText / top-8",
    "qwen25_objective_all_layers_1k_v1": "Qwen / WikiText / all-layer / 8 prompts",
    "llama_objective_1k_v1": "Llama / WikiText / top-8",
    "olmo2_objective_1k_v1": "OLMo / WikiText / top-8",
    "llama_all_layers_objective_b6_v1": "Llama / WikiText / all-layer / b6",
    "olmo2_all_layers_objective_b6_v1": "OLMo / WikiText / all-layer / b6",
    "qwen25_all_layers_objective_b3_v1": "Qwen / WikiText / all-layer / b3",
}


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def matrix_label(name: str) -> str:
    return MATRIX_LABELS.get(name, name.replace("_", " "))


def discover_aggregates(
    results_root: Path,
    *,
    include_incomplete: bool,
    include_smoke: bool,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    accepted = []
    rejected = []
    for path in sorted(results_root.glob("*/aggregate/summary.json")):
        matrix = path.parent.parent.name
        if not include_smoke and "smoke" in matrix:
            continue
        summary = read_json(path)
        if "grouped" not in summary or "cross_context_effects" not in summary:
            continue
        missing = int(summary.get("num_missing_pairs", 0))
        stale = int(summary.get("num_rejected_pairs", 0))
        complete = missing == 0 and stale == 0
        record = {
            "matrix": matrix,
            "label": matrix_label(matrix),
            "path": str(path),
            "complete": complete,
            "missing_pairs": missing,
            "rejected_pairs": stale,
            "summary": summary,
        }
        if complete or include_incomplete:
            accepted.append(record)
        else:
            rejected.append({key: value for key, value in record.items() if key != "summary"})
    return accepted, rejected


def discover_final_results(results_root: Path, *, include_smoke: bool) -> List[Dict[str, Any]]:
    records = []
    for path in sorted(results_root.glob("*/final_results/summary.json")):
        campaign = path.parent.parent.name
        if not include_smoke and "smoke" in campaign:
            continue
        summary = read_json(path)
        if not summary.get("rows") or not summary.get("baseline"):
            continue
        versions = summary.get("evaluator_versions", {})
        records.append(
            {
                "campaign": campaign,
                "label": FINAL_LABELS.get(campaign, matrix_label(campaign)),
                "path": str(path),
                "valid_evaluators": versions.get("quality") == "teacher_forced_cached_v1"
                and versions.get("acceptance") == "cached_dynamic_v4",
                "summary": summary,
            }
        )
    return records


def mean_saving_by_budget(grouped: Iterable[Dict[str, Any]]) -> Dict[int, float]:
    values: Dict[int, List[float]] = {}
    for row in grouped:
        if row.get("allocation_objective") != "quality":
            continue
        budget = int(row["budget"])
        values.setdefault(budget, []).append(float(row["total_cache_saved_fraction"]))
    return {budget: mean(items) for budget, items in values.items()}


def collect_rows(records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped_rows = []
    objective_rows = []
    kv_rows = []
    native_rows = []
    exactness_rows = []
    for record in records:
        summary = record["summary"]
        prefix = {
            "matrix": record["matrix"],
            "matrix_label": record["label"],
            "complete": record["complete"],
        }
        grouped = summary.get("grouped", [])
        savings = mean_saving_by_budget(grouped)
        grouped_rows.extend({**prefix, **row} for row in grouped)
        for row in summary.get("cross_context_effects", []):
            budget = int(row["budget"])
            objective_rows.append(
                {
                    **prefix,
                    **row,
                    "total_cache_saved_fraction": savings.get(budget, float("nan")),
                }
            )
        for row in summary.get("kv_priority_cross_context_effects", []):
            budget = int(row["budget"])
            kv_rows.append(
                {
                    **prefix,
                    **row,
                    "total_cache_saved_fraction": savings.get(budget, float("nan")),
                }
            )
        for row in summary.get("native_acceptance_cross_context_effects", []):
            budget = int(row["budget"])
            native_rows.append(
                {
                    **prefix,
                    **row,
                    "total_cache_saved_fraction": savings.get(budget, float("nan")),
                }
            )
        totals = summary.get("exactness_audit", {}).get("totals", {})
        exact = int(totals.get("exact", 0))
        ties = int(totals.get("numerical_tie", 0))
        non_tie = int(totals.get("non_tie_or_unknown", 0))
        checks = exact + ties + non_tie
        exactness_rows.append(
            {
                **prefix,
                "exact": exact,
                "numerical_tie": ties,
                "non_tie_or_unknown": non_tie,
                "invalid_prompts": int(totals.get("invalid_prompts", 0)),
                "total_checks": checks,
                "exact_or_tie_fraction": (exact + ties) / checks if checks else float("nan"),
            }
        )
    return {
        "grouped": grouped_rows,
        "objective": objective_rows,
        "kv": kv_rows,
        "native": native_rows,
        "exactness": exactness_rows,
    }


def collect_final_rows(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for record in records:
        summary = record["summary"]
        baseline = summary["baseline"]
        audit = summary.get("acceptance_exactness_audit", {})
        audit_effects = audit.get("effects", {})
        effect_by_allocation = {
            "quality_optimized": audit_effects.get("quality_vs_native"),
            "acceptance_optimized": audit_effects.get("acceptance_vs_native"),
        }
        for row in summary["rows"]:
            flattened = {
                "campaign": record["campaign"],
                "campaign_label": record["label"],
                "valid_evaluators": record["valid_evaluators"],
                "baseline_quality_nll": baseline.get("quality_nll"),
                "baseline_spec_accept_rate": baseline.get("spec_accept_rate"),
                **row,
            }
            effect = effect_by_allocation.get(str(row.get("allocation")))
            if effect:
                flattened["spec_accept_rate_delta_raw"] = row.get("spec_accept_rate_delta")
                flattened["spec_accept_rate_delta"] = effect.get("mean")
                flattened["spec_accept_rate_delta_ci_low"] = effect.get("ci_low")
                flattened["spec_accept_rate_delta_ci_high"] = effect.get("ci_high")
                flattened["exactness_valid_prompts"] = audit.get("valid_prompts")
                flattened["exactness_excluded_prompts"] = audit.get("excluded_non_tie_prompts")
            rows.append(flattened)
    return rows


def configure_plot_style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.2,
        }
    )


def save_figure(fig: Any, out_dir: Path, stem: str) -> List[str]:
    paths = []
    for extension in ("png", "pdf"):
        path = out_dir / f"{stem}.{extension}"
        fig.savefig(path, dpi=240, bbox_inches="tight")
        paths.append(str(path))
    return paths


def plot_objective_effects(rows: List[Dict[str, Any]], out_dir: Path) -> List[str]:
    if not rows:
        return []
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    fig, axis = plt.subplots(figsize=(9.5, 5.0))
    for label in sorted({str(row["matrix_label"]) for row in rows}):
        subset = sorted(
            [row for row in rows if row["matrix_label"] == label and finite(row.get("paired_acceptance_mean"))],
            key=lambda row: int(row["budget"]),
        )
        if not subset:
            continue
        y = [100.0 * float(row["paired_acceptance_mean"]) for row in subset]
        low = [100.0 * (float(row["paired_acceptance_mean"]) - float(row["paired_acceptance_ci_low"])) for row in subset]
        high = [100.0 * (float(row["paired_acceptance_ci_high"]) - float(row["paired_acceptance_mean"])) for row in subset]
        axis.errorbar(
            [int(row["budget"]) for row in subset],
            y,
            yerr=[low, high],
            marker="o",
            linewidth=1.8,
            capsize=3,
            label=label,
        )
    axis.axhline(0.0, color="#222222", linewidth=1)
    axis.set_xlabel("Mean KV bit budget")
    axis.set_ylabel("Acceptance-aware minus quality-aware acceptance (points)")
    axis.set_title("Downstream-Objective Effect Under Matched KV Memory", fontweight="bold")
    axis.legend(fontsize=7, frameon=False, ncol=2)
    fig.tight_layout()
    paths = save_figure(fig, out_dir, "objective_acceptance_effects")
    plt.close(fig)
    return paths


def plot_kv_effects(rows: List[Dict[str, Any]], out_dir: Path) -> List[str]:
    usable = [row for row in rows if finite(row.get("paired_acceptance_mean")) and finite(row.get("paired_quality_kl_mean"))]
    if not usable:
        return []
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    labels = [f"{row['matrix_label']} / b{row['budget']}" for row in usable]
    y = list(range(len(usable)))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, max(4.0, 0.42 * len(usable) + 1.5)), sharey=True)
    axes[0].barh(y, [100.0 * float(row["paired_acceptance_mean"]) for row in usable], color="#D1495B")
    axes[0].axvline(0.0, color="#222222", linewidth=1)
    axes[0].set_xlabel("K-priority acceptance advantage (points)")
    axes[1].barh(y, [float(row["paired_quality_kl_mean"]) for row in usable], color="#2A9D8F")
    axes[1].axvline(0.0, color="#222222", linewidth=1)
    axes[1].set_xlabel("K-priority KL advantage")
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    fig.suptitle("Equal-Memory K-Priority vs V-Priority Quantization", fontweight="bold")
    fig.tight_layout()
    paths = save_figure(fig, out_dir, "kv_priority_effects")
    plt.close(fig)
    return paths


def plot_final_results(rows: List[Dict[str, Any]], out_dir: Path) -> List[str]:
    usable = [
        row
        for row in rows
        if row.get("valid_evaluators")
        and finite(row.get("total_cache_saved_fraction"))
        and finite(row.get("spec_accept_rate_delta"))
    ]
    if not usable:
        return []
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    colors = {"quality_optimized": "#274466", "acceptance_optimized": "#D1495B"}
    markers = {"quality_optimized": "o", "acceptance_optimized": "s"}
    fig, axis = plt.subplots(figsize=(9.5, 5.4))
    for allocation in ("quality_optimized", "acceptance_optimized"):
        selected = [row for row in usable if row.get("allocation") == allocation]
        axis.scatter(
            [100.0 * float(row["total_cache_saved_fraction"]) for row in selected],
            [100.0 * float(row["spec_accept_rate_delta"]) for row in selected],
            s=65,
            marker=markers[allocation],
            color=colors[allocation],
            label=allocation.replace("_", " "),
            zorder=3,
        )
        for row in selected:
            axis.annotate(
                str(row["campaign_label"]),
                (
                    100.0 * float(row["total_cache_saved_fraction"]),
                    100.0 * float(row["spec_accept_rate_delta"]),
                ),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=7,
            )
    axis.axhline(0.0, color="#222222", linewidth=1)
    axis.set_xlabel("Total speculative KV cache saved (%)")
    axis.set_ylabel("Acceptance change from native draft (percentage points)")
    axis.set_title("All-Layer Memory-Acceptance Tradeoff", fontweight="bold")
    axis.legend(frameon=False)
    fig.tight_layout()
    paths = save_figure(fig, out_dir, "all_layer_memory_acceptance")
    plt.close(fig)
    return paths


def plot_native_acceptance(rows: List[Dict[str, Any]], out_dir: Path) -> List[str]:
    usable = [
        row
        for row in rows
        if finite(row.get("total_cache_saved_fraction"))
        and finite(row.get("paired_acceptance_mean"))
        and finite(row.get("paired_acceptance_ci_low"))
        and finite(row.get("paired_acceptance_ci_high"))
    ]
    if not usable:
        return []
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    colors = {
        "quality": "#26456E",
        "acceptance": "#D1495B",
        "k_priority": "#2A9D8F",
        "v_priority": "#E9C46A",
    }
    fig, axis = plt.subplots(figsize=(10.0, 5.6))
    for objective in ("quality", "acceptance", "k_priority", "v_priority"):
        selected = [row for row in usable if row.get("allocation_objective") == objective]
        if not selected:
            continue
        x = [100.0 * float(row["total_cache_saved_fraction"]) for row in selected]
        y = [100.0 * float(row["paired_acceptance_mean"]) for row in selected]
        low = [
            100.0 * (float(row["paired_acceptance_mean"]) - float(row["paired_acceptance_ci_low"]))
            for row in selected
        ]
        high = [
            100.0 * (float(row["paired_acceptance_ci_high"]) - float(row["paired_acceptance_mean"]))
            for row in selected
        ]
        axis.errorbar(
            x,
            y,
            yerr=[low, high],
            fmt="o",
            capsize=3,
            color=colors[objective],
            label=objective.replace("_", " "),
        )
    axis.axhline(0.0, color="#222222", linewidth=1)
    axis.set_xlabel("Total speculative KV cache saved (%)")
    axis.set_ylabel("Acceptance change from native draft (percentage points)")
    axis.set_title("Acceptance Retention Under KV Quantization", fontweight="bold")
    axis.legend(frameon=False, ncol=2)
    fig.tight_layout()
    paths = save_figure(fig, out_dir, "native_acceptance_retention")
    plt.close(fig)
    return paths


def latex_escape(value: Any) -> str:
    return str(value).replace("_", r"\_").replace("%", r"\%")


def fmt_ci(row: Dict[str, Any], metric: str, *, scale: float = 1.0, digits: int = 3) -> str:
    if not finite(row.get(f"paired_{metric}_mean")):
        return "--"
    estimate = scale * float(row[f"paired_{metric}_mean"])
    low = scale * float(row[f"paired_{metric}_ci_low"])
    high = scale * float(row[f"paired_{metric}_ci_high"])
    return f"{estimate:+.{digits}f} [{low:+.{digits}f}, {high:+.{digits}f}]"


def write_latex_tables(
    rows: Dict[str, List[Dict[str, Any]]], final_rows: List[Dict[str, Any]], out_dir: Path
) -> List[str]:
    objective_lines = [
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"Matrix & Bits & $\Delta$ acceptance (pp, 95\% CI) & Total KV saved \\",
        r"\midrule",
    ]
    for row in rows["objective"]:
        saved = 100.0 * float(row["total_cache_saved_fraction"])
        objective_lines.append(
            f"{latex_escape(row['matrix_label'])} & {row['budget']} & "
            f"{fmt_ci(row, 'acceptance', scale=100.0, digits=2)} & {saved:.1f}\\% \\\\"
        )
    objective_lines.extend([r"\bottomrule", r"\end{tabular}"])
    objective_path = out_dir / "objective_campaign_table.tex"
    objective_path.write_text("\n".join(objective_lines) + "\n", encoding="utf-8")

    kv_lines = [
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"Matrix & Bits & K-priority $\Delta$ acceptance (pp) & K-priority $\Delta$ KL \\",
        r"\midrule",
    ]
    for row in rows["kv"]:
        kv_lines.append(
            f"{latex_escape(row['matrix_label'])} & {row['budget']} & "
            f"{fmt_ci(row, 'acceptance', scale=100.0, digits=2)} & "
            f"{fmt_ci(row, 'quality_kl', digits=4)} \\\\" 
        )
    kv_lines.extend([r"\bottomrule", r"\end{tabular}"])
    kv_path = out_dir / "kv_priority_table.tex"
    kv_path.write_text("\n".join(kv_lines) + "\n", encoding="utf-8")

    native_lines = [
        r"\begin{tabular}{lllrr}",
        r"\toprule",
        r"Matrix & Bits & Allocation & $\Delta$ acceptance vs native (pp) & Total KV saved \\",
        r"\midrule",
    ]
    for row in rows["native"]:
        saved = 100.0 * float(row["total_cache_saved_fraction"])
        native_lines.append(
            f"{latex_escape(row['matrix_label'])} & {row['budget']} & "
            f"{latex_escape(row['allocation_objective'])} & "
            f"{fmt_ci(row, 'acceptance', scale=100.0, digits=2)} & "
            f"{saved:.1f}\\% \\\\"
        )
    native_lines.extend([r"\bottomrule", r"\end{tabular}"])
    native_path = out_dir / "native_acceptance_table.tex"
    native_path.write_text("\n".join(native_lines) + "\n", encoding="utf-8")
    final_lines = [
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Campaign & Allocation & Mean bits & Total KV saved & $\Delta$ acceptance (pp) \\",
        r"\midrule",
    ]
    for row in final_rows:
        if not row.get("valid_evaluators"):
            continue
        final_lines.append(
            f"{latex_escape(row['campaign_label'])} & {latex_escape(row['allocation'])} & "
            f"{float(row['all_component_mean_bits']):.1f} & "
            f"{100.0 * float(row['total_cache_saved_fraction']):.1f}\\% & "
            f"{100.0 * float(row['spec_accept_rate_delta']):+.2f} \\\\"
        )
    final_lines.extend([r"\bottomrule", r"\end{tabular}"])
    final_path = out_dir / "objective_final_results_table.tex"
    final_path.write_text("\n".join(final_lines) + "\n", encoding="utf-8")
    return [str(objective_path), str(kv_path), str(native_path), str(final_path)]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_root", type=Path, default=Path("outputs/objective_kv"))
    parser.add_argument("--out_dir", type=Path, default=Path("paper/objective_campaign_artifacts"))
    parser.add_argument("--include_incomplete", action="store_true")
    parser.add_argument("--include_smoke", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    records, rejected = discover_aggregates(
        args.results_root,
        include_incomplete=args.include_incomplete,
        include_smoke=args.include_smoke,
    )
    final_records = discover_final_results(args.results_root, include_smoke=args.include_smoke)
    rows = collect_rows(records)
    final_rows = collect_final_rows(final_records)
    for name, values in rows.items():
        write_csv(args.out_dir / f"objective_campaign_{name}.csv", values)
    write_csv(args.out_dir / "objective_campaign_final_results.csv", final_rows)
    write_csv(args.out_dir / "objective_campaign_rejected.csv", rejected)

    plots = []
    try:
        configure_plot_style()
        plots.extend(plot_objective_effects(rows["objective"], args.out_dir))
        plots.extend(plot_kv_effects(rows["kv"], args.out_dir))
        plots.extend(plot_native_acceptance(rows["native"], args.out_dir))
        plots.extend(plot_final_results(final_rows, args.out_dir))
    except ImportError:
        pass
    tables = write_latex_tables(rows, final_rows, args.out_dir)
    payload = {
        "results_root": str(args.results_root),
        "num_complete_matrices": sum(bool(record["complete"]) for record in records),
        "num_included_matrices": len(records),
        "included_matrices": [
            {key: value for key, value in record.items() if key != "summary"} for record in records
        ],
        "rejected_matrices": rejected,
        "included_final_results": [
            {key: value for key, value in record.items() if key != "summary"}
            for record in final_records
        ],
        "row_counts": {name: len(values) for name, values in rows.items()},
        "num_final_result_rows": len(final_rows),
        "plots": plots,
        "tables": tables,
    }
    (args.out_dir / "objective_campaign_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Included {len(records)} matrices ({payload['num_complete_matrices']} complete); "
        f"rejected {len(rejected)} incomplete matrices."
    )
    print(f"Artifacts: {args.out_dir}")


if __name__ == "__main__":
    main()
