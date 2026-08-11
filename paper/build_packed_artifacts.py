#!/usr/bin/env python3
"""Build paper artifacts from the actual bit-packed KV attention benchmark."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt


PAPER_DIR = Path(__file__).resolve().parent
SOURCE = PAPER_DIR / "data" / "packed" / "qwen25_15b_summary.json"
FUSED_SOURCE = PAPER_DIR / "data" / "packed" / "qwen25_15b_fused_split512_summary.json"
OUT_DIR = PAPER_DIR / "packed_artifacts"
EXPECTED_VERSION = "packed_unfused_attention_v1"
EXPECTED_FUSED_VERSION = "packed_fused_attention_v1"
COLORS = {
    "none": "#17324D",
    "k8v4": "#168C82",
    "k4v8": "#D1495B",
    "k4v4": "#667085",
    "k3v4": "#6E56CF",
    "k4v3": "#D99A2B",
}
LABELS = {
    "none": "BF16",
    "k8v4": "K8V4",
    "k4v8": "K4V8",
    "k4v4": "K4V4",
    "k3v4": "K3V4",
    "k4v3": "K4V3",
}


def load_rows() -> List[Dict[str, Any]]:
    payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    runtime = payload.get("runtime", {})
    if runtime.get("evaluator_version") != EXPECTED_VERSION:
        raise ValueError("Packed benchmark provenance is missing or unsupported.")
    if runtime.get("storage_mode") != "actual_bit_packed_uint8_payloads":
        raise ValueError("Packed benchmark did not use actual packed payloads.")
    if runtime.get("production_throughput_claim") is not False:
        raise ValueError("Unfused benchmark must not be marked as production throughput.")
    return payload["rows"]


def load_fused_rows() -> List[Dict[str, Any]]:
    payload = json.loads(FUSED_SOURCE.read_text(encoding="utf-8"))
    runtime = payload.get("runtime", {})
    if runtime.get("evaluator_version") != EXPECTED_FUSED_VERSION:
        raise ValueError("Direct packed benchmark provenance is missing or unsupported.")
    if runtime.get("storage_mode") != "actual_bit_packed_uint8_payloads":
        raise ValueError("Direct packed benchmark did not use actual packed payloads.")
    if runtime.get("decode_mode") != "triton_online_softmax_direct_packed_kv":
        raise ValueError("Direct packed benchmark used an unexpected decode path.")
    if runtime.get("production_throughput_claim") is not False:
        raise ValueError("Microbenchmark must not be marked as production throughput.")
    return payload["rows"]


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.frameon": False,
            "figure.dpi": 180,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )


def plot(rows: List[Dict[str, Any]]) -> None:
    contexts = sorted({int(row["context"]) for row in rows})
    by_key = {(int(row["context"]), row["config"]): row for row in rows}
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.55))

    for config in ("none", "k4v8", "k8v4", "k4v4", "k3v4", "k4v3"):
        axes[0].plot(
            contexts,
            [by_key[(context, config)]["packed_cache_bytes_model"] / 2**20 for context in contexts],
            marker="o",
            markersize=3.5,
            linewidth=1.5,
            color=COLORS[config],
            label=LABELS[config],
        )
        if config != "none":
            axes[1].plot(
                contexts,
                [100.0 * by_key[(context, config)]["cache_saved_fraction"] for context in contexts],
                marker="o",
                markersize=3.5,
                linewidth=1.5,
                color=COLORS[config],
                label=LABELS[config],
            )

    for axis in axes:
        axis.set_xscale("log", base=2)
        axis.set_xticks(contexts, [f"{context // 1024}K" for context in contexts])
        axis.grid(color="#D9DDD8", linewidth=0.7, alpha=0.8)
        axis.set_axisbelow(True)
        axis.set_xlabel("Context length")
    axes[0].set_yscale("log", base=2)
    axes[0].set_ylabel("Actual persistent KV storage (MiB)")
    axes[0].set_title("Bit-packed cache footprint", fontweight="bold")
    axes[1].set_ylabel("KV storage saved vs. BF16 (%)")
    axes[1].set_ylim(50, 77)
    axes[1].set_title("Metadata-inclusive savings", fontweight="bold")
    axes[0].legend(ncol=2, fontsize=7.5)
    axes[1].legend(ncol=2, fontsize=7.5)
    fig.tight_layout(w_pad=1.6)
    fig.savefig(OUT_DIR / "packed_kv_memory.pdf")
    fig.savefig(OUT_DIR / "packed_kv_memory.png")
    plt.close(fig)


def plot_kernel(fused_rows: List[Dict[str, Any]]) -> None:
    rows = sorted(
        (row for row in fused_rows if row["config"] == "k4v4"),
        key=lambda row: int(row["context"]),
    )
    contexts = [int(row["context"]) for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.55))

    latency_series = (
        ("Native BF16 SDPA", "native_decode_median_ms", COLORS["none"]),
        ("Materialize then attend", "unfused_decode_median_ms", COLORS["k4v8"]),
        ("Direct packed Triton", "fused_decode_median_ms", COLORS["k8v4"]),
    )
    for label, field, color in latency_series:
        axes[0].plot(
            contexts,
            [row[field] for row in rows],
            marker="o",
            markersize=3.5,
            linewidth=1.5,
            color=color,
            label=label,
        )

    axes[1].plot(
        contexts,
        [row["unfused_transient_peak_delta_bytes"] / 2**20 for row in rows],
        marker="o",
        markersize=3.5,
        linewidth=1.5,
        color=COLORS["k4v8"],
        label="Materialize then attend",
    )
    axes[1].plot(
        contexts,
        [row["fused_transient_peak_delta_bytes"] / 2**20 for row in rows],
        marker="o",
        markersize=3.5,
        linewidth=1.5,
        color=COLORS["k8v4"],
        label="Direct packed Triton",
    )

    for axis in axes:
        axis.set_xscale("log", base=2)
        axis.set_yscale("log", base=2)
        axis.set_xticks(contexts, [f"{context // 1024}K" for context in contexts])
        axis.grid(color="#D9DDD8", linewidth=0.7, alpha=0.8)
        axis.set_axisbelow(True)
        axis.set_xlabel("Context length")
        axis.legend(fontsize=7.2)
    axes[0].set_ylabel("Single-layer decode latency (ms)")
    axes[0].set_title("K4V4 attention microbenchmark", fontweight="bold")
    axes[1].set_ylabel("Temporary allocation (MiB)")
    axes[1].set_title("Dequantization workspace", fontweight="bold")
    fig.tight_layout(w_pad=1.6)
    fig.savefig(OUT_DIR / "packed_kv_kernel.pdf")
    fig.savefig(OUT_DIR / "packed_kv_kernel.png")
    plt.close(fig)


def write_outputs(rows: List[Dict[str, Any]]) -> None:
    fields = [
        "context",
        "config",
        "native_cache_bytes_model",
        "packed_cache_bytes_model",
        "cache_saved_fraction",
        "unfused_slowdown_vs_native",
        "output_cosine",
        "output_relative_rmse",
    ]
    with (OUT_DIR / "packed_kv_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})

    by_key = {(int(row["context"]), row["config"]): row for row in rows}
    table = [
        r"\begin{tabular}{rrrrr}",
        r"\toprule",
        r"Context & BF16 MiB & K4V4 MiB & Saved (\%) & Unfused slowdown \\",
        r"\midrule",
    ]
    for context in sorted({int(row["context"]) for row in rows}):
        native = by_key[(context, "none")]
        compressed = by_key[(context, "k4v4")]
        table.append(
            f"{context // 1024}K & {native['native_cache_bytes_model'] / 2**20:.1f} & "
            f"{compressed['packed_cache_bytes_model'] / 2**20:.1f} & "
            f"{100.0 * compressed['cache_saved_fraction']:.2f} & "
            f"{compressed['unfused_slowdown_vs_native']:.1f}$\\times$ \\\\"
        )
    table.extend([r"\bottomrule", r"\end{tabular}"])
    (OUT_DIR / "packed_kv_table.tex").write_text("\n".join(table) + "\n", encoding="utf-8")


def write_kernel_outputs(fused_rows: List[Dict[str, Any]]) -> None:
    rows = sorted(
        (row for row in fused_rows if row["config"] == "k4v4"),
        key=lambda row: int(row["context"]),
    )
    fields = [
        "context",
        "native_decode_median_ms",
        "unfused_decode_median_ms",
        "fused_decode_median_ms",
        "fused_speedup_vs_unfused",
        "fused_speedup_vs_native",
        "unfused_transient_peak_delta_bytes",
        "fused_transient_peak_delta_bytes",
        "kernel_output_cosine",
        "kernel_output_relative_rmse",
    ]
    with (OUT_DIR / "packed_kv_kernel_results.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})

    table = [
        r"\begin{tabular}{rrrrrr}",
        r"\toprule",
        r"Context & Native ms & Unfused ms & Direct ms & Direct speedup & Cosine \\",
        r"\midrule",
    ]
    for row in rows:
        table.append(
            f"{int(row['context']) // 1024}K & {row['native_decode_median_ms']:.3f} & "
            f"{row['unfused_decode_median_ms']:.3f} & {row['fused_decode_median_ms']:.3f} & "
            f"{row['fused_speedup_vs_unfused']:.2f}$\\times$ & "
            f"{row['kernel_output_cosine']:.6f} \\\\"
        )
    table.extend([r"\bottomrule", r"\end{tabular}"])
    (OUT_DIR / "packed_kv_kernel_table.tex").write_text(
        "\n".join(table) + "\n", encoding="utf-8"
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    fused_rows = load_fused_rows()
    configure_style()
    plot(rows)
    plot_kernel(fused_rows)
    write_outputs(rows)
    write_kernel_outputs(fused_rows)
    print(f"Wrote packed-cache artifacts to {OUT_DIR}.")


if __name__ == "__main__":
    main()
