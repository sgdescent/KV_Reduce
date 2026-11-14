"""Regenerate Figure 2 (geometry_interaction.pdf); CPU only.

Run from the repository root:

    python paper/figures/geometry_interaction.py

Panel (a) reads the frozen K8V4 - K4V8 contrasts behind Table 3. Panel (b)
plots the Qwen2.5-1.5B passkey counts of Table 5 and Appendix C; they are
copied here because the passkey aggregate is not yet frozen in ``results/``.
Layout, colors, and font sizes follow the original figure; pair colors match
Figure 3.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
GEOMETRY = ROOT / "results" / "cross-family" / "quantizer_geometry_summary.json"
OUT = Path(__file__).resolve().with_suffix(".pdf")

# pair key in the summary, legend label, color (same as Figure 3)
PAIRS = [
    ("llama31_8b_llama32_3b", "Llama 3.1/3.2", "#d99a2b"),
    ("olmo2_7b_1b", "OLMo 2 7B/1B", "#8a6fa8"),
    ("qwen25_3b_15b", "Qwen2.5 3B/1.5B", "#17324d"),
    ("qwen25_7b_3b", "Qwen2.5 7B/3B", "#3d6d9a"),
    ("qwen3_8b_4b", "Qwen3 8B/4B", "#188b7a"),
    ("smollm2_17b_360m", "SmolLM2 1.7B/360M", "#c84b5a"),
]
# Qwen2.5-1.5B 16K passkey: correct retrievals out of 192 (Table 5, App. C)
PASSKEY = {"K4V2": (166, 189), "K2V4": (10, 158)}  # (per-token, grouped) keys
N_PASSKEY = 192
NAVY, RED, GOLD, GRID = "#17324d", "#c84b5a", "#d99a2b", "#d9ddd8"
BOX = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.92)
MINUS = "\N{MINUS SIGN}"


def style_axes(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=7.2)
    ax.set_axisbelow(True)


def geometry_shift(ax) -> list:
    summary = json.loads(GEOMETRY.read_text(encoding="utf-8"))
    rows = {row["pair"]: row for row in summary["rows"]}
    ax.axhspan(-1, 1, color="#dcefe8", alpha=0.65, zorder=0)
    ax.grid(axis="y", color=GRID, linewidth=0.65)
    ax.axhline(0, color="#111111", linewidth=0.8)
    handles = []
    for key, label, color in PAIRS:
        row = rows[key]
        ys = [100 * row["naive_k8v4_minus_k4v8"], 100 * row["kivi_k8v4_minus_k4v8"]]
        (line,) = ax.plot([0, 1], ys, "-o", color=color, linewidth=1.5,
                          markersize=4.5, alpha=0.95, label=label)
        handles.append(line)
    macro = [100 * summary["naive_macro"]["mean"], 100 * summary["kivi_macro"]["mean"]]
    (line,) = ax.plot([0, 1], macro, "--*", color="#111111", linewidth=2.0,
                      markersize=8, zorder=4, label="Equal-model macro")
    handles.append(line)
    ax.text(0.62, 25.8, "all six grouped-key effects\nland in the shaded ±1 pp band",
            color="#188b7a", fontsize=6.6, ha="center", va="center", bbox=BOX,
            zorder=3)
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-4.2, 34.0)
    ax.set_xticks([0, 1], ["Per-token\nsymmetric", "Grouped\nper-channel keys"])
    ax.set_ylabel(f"Acceptance: K8V4 {MINUS} K4V8 (pp)", fontsize=8)
    ax.set_title("(a) Same models, same bits, new quantizer", loc="left",
                 fontsize=9, fontweight="bold")
    style_axes(ax)
    return handles


def retrieval_interaction(ax) -> None:
    k4v2 = [100 * n / N_PASSKEY for n in PASSKEY["K4V2"]]
    k2v4 = [100 * n / N_PASSKEY for n in PASSKEY["K2V4"]]
    ax.grid(axis="y", color=GRID, linewidth=0.65)
    ax.fill_between([0, 1], k2v4, k4v2, color=GOLD, alpha=0.14, linewidth=1.0)
    ax.plot([0, 1], k4v2, "-o", color=NAVY, linewidth=2.2, markersize=5.5)
    ax.plot([0, 1], k2v4, "-o", color=RED, linewidth=2.2, markersize=5.5)
    ax.text(-0.05, 94.8, f"{k4v2[0]:.1f}", color=NAVY, fontsize=7,
            ha="center", va="center", bbox=BOX)
    ax.text(-0.05, 13.5, f"{k2v4[0]:.1f}", color=RED, fontsize=7,
            ha="center", va="center", bbox=BOX)
    ax.text(0.0, 47.0, f"{k4v2[0] - k2v4[0]:.2f} pp gap", color="#8a5a00",
            fontsize=6.5, ha="center", va="center", bbox=BOX)
    ax.text(0.73, 91.0, f"{k4v2[1] - k2v4[1]:.2f} pp gap", color="#8a5a00",
            fontsize=6.5, ha="center", va="center", bbox=BOX)
    ax.text(1.07, 105.1, f"K4V2  {k4v2[1]:.1f}%", color=NAVY, fontsize=8,
            ha="left", va="center", bbox=BOX, clip_on=False)
    ax.text(1.07, 75.6, f"K2V4  {k2v4[1]:.1f}%", color=RED, fontsize=8,
            ha="left", va="center", bbox=BOX, clip_on=False)
    ax.set_xlim(-0.12, 1.30)
    ax.set_ylim(0, 106)
    ax.set_xticks([0, 1], ["Per-token\nkeys", "Grouped\nper-channel keys"])
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("16K passkey accuracy (%)", fontsize=8)
    ax.set_title("(b) Retrieval interaction", loc="left", fontsize=9,
                 fontweight="bold")
    style_axes(ax)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    width, height = 480.007, 193.563  # points, the size of the original figure
    fig = plt.figure(figsize=(width / 72, height / 72))
    axes = [fig.add_axes([x0 / width, 23.06 / height, (x1 - x0) / width, 119.4 / height])
            for x0, x1 in ((56.6, 260.5), (308.8, 450.0))]
    handles = geometry_shift(axes[0])
    retrieval_interaction(axes[1])
    fig.legend(handles=handles, loc="upper left",
               bbox_to_anchor=(0.0, 0.998), ncol=4, frameon=False, fontsize=6.5,
               handlelength=1.6, handletextpad=0.35, columnspacing=0.9,
               labelspacing=0.57, borderaxespad=0.2)
    fig.savefig(args.out, metadata={"CreationDate": None})
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
