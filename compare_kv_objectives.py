#!/usr/bin/env python3
"""Compare ordinary-quality and speculative-acceptance KV sensitivity maps."""

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from kv_utils import write_json


def read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows: List[Dict[str, Any]], path: str) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def candidate_key(row: Dict[str, str]) -> Tuple[int, str, int]:
    return int(float(row["layer"])), row["component"], int(float(row["bits"]))


def ranks(values: Sequence[float]) -> List[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    out = [0.0] * len(values)
    start = 0
    while start < len(indexed):
        end = start + 1
        while end < len(indexed) and indexed[end][1] == indexed[start][1]:
            end += 1
        average_rank = 0.5 * (start + end - 1) + 1.0
        for position in range(start, end):
            out[indexed[position][0]] = average_rank
        start = end
    return out


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_scale = math.sqrt(sum((x - x_mean) ** 2 for x in xs))
    y_scale = math.sqrt(sum((y - y_mean) ** 2 for y in ys))
    if x_scale == 0.0 or y_scale == 0.0:
        return float("nan")
    return numerator / (x_scale * y_scale)


def top_keys(rows: Sequence[Dict[str, Any]], field: str, count: int) -> set[Tuple[int, str, int]]:
    ordered = sorted(rows, key=lambda row: float(row[field]), reverse=True)
    return {(int(row["layer"]), str(row["component"]), int(row["bits"])) for row in ordered[:count]}


def read_allocation(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def allocation_disagreement(first: Dict[str, Any], second: Dict[str, Any]) -> Dict[str, float]:
    first_k, first_v = first["k_bits"], first["v_bits"]
    second_k, second_v = second["k_bits"], second["v_bits"]
    if len(first_k) != len(second_k) or len(first_v) != len(second_v):
        raise ValueError("Allocation layer counts differ; compare objectives on the same model.")
    pairs = list(zip(first_k, second_k)) + list(zip(first_v, second_v))
    disagreements = sum(int(int(a) != int(b)) for a, b in pairs)
    return {
        "component_decisions": float(len(pairs)),
        "component_disagreements": float(disagreements),
        "allocation_disagreement_fraction": disagreements / max(1, len(pairs)),
        "quality_mean_bits": sum(map(int, first_k + first_v)) / max(1, len(pairs)),
        "acceptance_mean_bits": sum(map(int, second_k + second_v)) / max(1, len(pairs)),
    }


def component_means(rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[f"{row['component']}{row['bits']}"].append(row)
    return {
        name: {
            "quality_risk": sum(float(row["quality_risk"]) for row in group) / len(group),
            "acceptance_risk": sum(float(row["acceptance_risk"]) for row in group) / len(group),
            "count": float(len(group)),
        }
        for name, group in grouped.items()
    }


def make_plots(rows: Sequence[Dict[str, Any]], out_dir: str) -> List[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    colors = {"k": "#26456E", "v": "#D1495B"}
    markers = {4: "o", 8: "s"}
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for row in rows:
        component = str(row["component"])
        bits = int(row["bits"])
        ax.scatter(
            float(row["quality_risk"]),
            float(row["acceptance_risk"]),
            color=colors.get(component, "#555555"),
            marker=markers.get(bits, "^"),
            s=55,
            alpha=0.85,
        )
    ax.set_xlabel("Ordinary LM quality risk (delta NLL)")
    ax.set_ylabel("Speculative risk (acceptance-rate drop)")
    ax.set_title("KV Precision Has Objective-Dependent Risk")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    paths = []
    for extension in ("png", "pdf"):
        path = os.path.join(out_dir, f"objective_sensitivity_scatter.{extension}")
        fig.savefig(path, dpi=220, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)

    layers = sorted({int(row["layer"]) for row in rows})
    bits_values = sorted({int(row["bits"]) for row in rows})
    fig, axes = plt.subplots(len(bits_values), 2, figsize=(10.2, max(3.4, 2.8 * len(bits_values))), squeeze=False)
    for row_idx, bits in enumerate(bits_values):
        for col_idx, (field, title) in enumerate(
            (("quality_risk", "Ordinary quality risk"), ("acceptance_risk", "Speculative acceptance risk"))
        ):
            matrix = []
            for component in ("k", "v"):
                lookup = {
                    int(row["layer"]): float(row[field])
                    for row in rows
                    if int(row["bits"]) == bits and row["component"] == component
                }
                matrix.append([lookup.get(layer, float("nan")) for layer in layers])
            image = axes[row_idx][col_idx].imshow(matrix, aspect="auto", cmap="magma")
            axes[row_idx][col_idx].set_title(f"{title}, {bits}-bit")
            axes[row_idx][col_idx].set_yticks([0, 1], labels=["K", "V"])
            axes[row_idx][col_idx].set_xticks(range(len(layers)), labels=layers, rotation=45)
            axes[row_idx][col_idx].set_xlabel("Layer")
            fig.colorbar(image, ax=axes[row_idx][col_idx], fraction=0.046, pad=0.04)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        path = os.path.join(out_dir, f"objective_sensitivity_heatmaps.{extension}")
        fig.savefig(path, dpi=220, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare ordinary-quality and SpecDec KV sensitivity.")
    parser.add_argument("--quality_profile_csv", type=str, required=True)
    parser.add_argument("--acceptance_profile_csv", type=str, required=True)
    parser.add_argument("--quality_allocation", type=str, default=None)
    parser.add_argument("--acceptance_allocation", type=str, default=None)
    parser.add_argument("--top_fraction", type=float, default=0.25)
    parser.add_argument("--out_dir", type=str, default="outputs/kv_objective_comparison")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    quality = {
        candidate_key(row): row
        for row in read_csv(args.quality_profile_csv)
        if row.get("component") in {"k", "v"}
    }
    acceptance = {
        candidate_key(row): row
        for row in read_csv(args.acceptance_profile_csv)
        if row.get("component") in {"k", "v"}
    }
    common = sorted(set(quality).intersection(acceptance))
    if not common:
        raise ValueError("No common layer/component/bit candidates were found.")

    rows: List[Dict[str, Any]] = []
    for layer, component, bits in common:
        quality_row = quality[(layer, component, bits)]
        acceptance_row = acceptance[(layer, component, bits)]
        rows.append(
            {
                "layer": layer,
                "component": component,
                "bits": bits,
                "quality_risk": max(0.0, float(quality_row.get("quality_risk") or quality_row["delta_nll"])),
                "delta_nll": float(quality_row["delta_nll"]),
                "quality_js": float(quality_row.get("js", 0.0)),
                "quality_top1_match": float(quality_row.get("top1_match", 0.0)),
                "acceptance_risk": max(0.0, float(acceptance_row["accept_rate_drop"])),
                "accept_rate": float(acceptance_row["accept_rate"]),
                "acceptance_js": float(acceptance_row.get("round_js", 0.0)),
                "acceptance_top1_match": float(acceptance_row.get("round_top1_match", 0.0)),
            }
        )

    quality_risks = [float(row["quality_risk"]) for row in rows]
    acceptance_risks = [float(row["acceptance_risk"]) for row in rows]
    top_count = max(1, round(len(rows) * args.top_fraction))
    quality_top = top_keys(rows, "quality_risk", top_count)
    acceptance_top = top_keys(rows, "acceptance_risk", top_count)
    union = quality_top.union(acceptance_top)
    summary: Dict[str, Any] = {
        "num_common_candidates": len(rows),
        "pearson_risk_correlation": pearson(quality_risks, acceptance_risks),
        "spearman_risk_correlation": pearson(ranks(quality_risks), ranks(acceptance_risks)),
        "top_fraction": args.top_fraction,
        "top_count": top_count,
        "sensitive_top_overlap_count": len(quality_top.intersection(acceptance_top)),
        "sensitive_top_jaccard": len(quality_top.intersection(acceptance_top)) / max(1, len(union)),
        "component_bit_means": component_means(rows),
    }
    if args.quality_allocation and args.acceptance_allocation:
        summary["allocation_comparison"] = allocation_disagreement(
            read_allocation(args.quality_allocation),
            read_allocation(args.acceptance_allocation),
        )

    plot_paths = make_plots(rows, args.out_dir)
    summary["plots"] = plot_paths
    write_csv(rows, os.path.join(args.out_dir, "joined_objective_sensitivity.csv"))
    write_json(summary, os.path.join(args.out_dir, "summary.json"))
    print("Done!")
    print(f"  spearman={summary['spearman_risk_correlation']:.4f}")
    print(f"  top_jaccard={summary['sensitive_top_jaccard']:.4f}")
    print(f"  {os.path.join(args.out_dir, 'summary.json')}")


if __name__ == "__main__":
    main()
