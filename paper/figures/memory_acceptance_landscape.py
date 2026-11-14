"""Regenerate Figure 3 (memory_acceptance_landscape.pdf); CPU only.

Run from the repository root:

    python paper/figures/memory_acceptance_landscape.py

Panel (a) plots the rows of Table 2 (total KV saved, K4V4 acceptance change and
paired 95% CI). They are copied here because the cross-family aggregate is not
yet frozen in ``results/``. Panel (b) reads the frozen packed-storage
measurements.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
PACKED = ROOT / "results" / "packed" / "qwen25_15b_summary.json"
OUT = Path(__file__).resolve().with_suffix(".pdf")

# label, total KV saved (%), K4V4 - BF16 acceptance (pp), 95% CI, color, marker
PAIRS = [
    ("Qwen2.5 3B/1.5B", 29.02, -0.76, (-2.23, 0.66), "#17324d", "o"),
    ("Qwen2.5 7B/3B", 25.96, 0.28, (-0.98, 1.47), "#3d6d9a", "s"),
    ("Qwen3 8B/4B", 33.17, -1.67, (-3.12, -0.30), "#188b7a", "^"),
    ("Llama 3.1/3.2", 30.96, 0.13, (-1.27, 1.60), "#d99a2b", "D"),
    ("OLMo 2 7B/1B", 13.27, -0.86, (-1.98, 0.18), "#8a6fa8", "P"),
    ("SmolLM2 1.7B/360M", 11.30, -1.85, (-3.21, -0.54), "#c84b5a", "X"),
]
MACRO = (23.95, -0.79, (-1.63, 0.04))  # equal-model macro row of Table 2
CONTEXTS = (1024, 4096, 16384, 32768)
NAVY, RED, GOLD, GRID = "#17324d", "#c84b5a", "#d99a2b", "#d9ddd8"
BOX = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.92)


def packed_mib(config: str) -> list[float]:
    rows = json.loads(PACKED.read_text(encoding="utf-8"))["rows"]
    by_context = {r["context"]: r for r in rows if r["config"] == config}
    return [by_context[c]["packed_cache_bytes_model"] / 2**20 for c in CONTEXTS]


def style_axes(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=7.2)
    ax.set_axisbelow(True)


def operating_region(ax):
    ax.axhspan(-2, 2, color="#dcefe8", alpha=0.55, zorder=0)
    ax.grid(axis="y", color=GRID, linewidth=0.65)
    ax.axhline(0, color="#111111", linewidth=0.8)
    handles = []
    for label, saved, delta, (low, high), color, marker in PAIRS:
        handles.append(ax.errorbar(
            saved, delta, yerr=[[delta - low], [high - delta]], fmt=marker,
            color=color, elinewidth=1.0, capsize=2.2, capthick=0.5,
            markersize=5.8, markeredgecolor="white", markeredgewidth=0.5,
            zorder=3, label=label))
    saved, delta, (low, high) = MACRO
    handles.append(ax.errorbar(
        saved, delta, yerr=[[delta - low], [high - delta]], fmt="*",
        color="#111111", elinewidth=1.4, capsize=2.5, capthick=0.5,
        markersize=10, markeredgecolor="white", markeredgewidth=0.5,
        zorder=4, label="Equal-model macro"))
    ax.annotate(f"macro\n{saved:.2f}% saved, {delta:.2f} pp", xy=(saved, delta),
                xytext=(18.0, -3.11), ha="center", va="center", fontsize=6.5,
                bbox=BOX, zorder=3, arrowprops=dict(
                    arrowstyle="-", color="#555555", linewidth=0.7,
                    connectionstyle="arc3,rad=0.5"))
    ax.text(9.1, 1.92, "central estimates within 2 pp", color="#188b7a",
            fontsize=6.4, ha="left", va="top")
    ax.set_xlim(8, 35.5)
    ax.set_ylim(-3.8, 2.3)
    ax.set_xticks(range(10, 36, 5))
    ax.set_xlabel("Total target + draft KV saved (%)", fontsize=8)
    ax.set_ylabel("K4V4 acceptance change (pp)", fontsize=8)
    ax.set_title("(a) Cross-family operating region", loc="left",
                 fontsize=9, fontweight="bold")
    style_axes(ax)
    return handles


def packed_savings(ax) -> None:
    bf16, k4v4 = packed_mib("none"), packed_mib("k4v4")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.grid(color=GRID, linewidth=0.65)
    ax.fill_between(CONTEXTS, k4v4, bf16, color=GOLD, alpha=0.18, linewidth=1.0)
    ax.plot(CONTEXTS, bf16, "-o", color=NAVY, linewidth=2.0, markersize=3.5)
    ax.plot(CONTEXTS, k4v4, "-o", color=RED, linewidth=2.0, markersize=3.5)
    ax.text(20000, 880, "BF16", color=NAVY, fontsize=8, ha="center",
            va="center", bbox=BOX)
    ax.text(19900, 179, "K4V4", color=RED, fontsize=8, ha="center",
            va="center", bbox=BOX)
    ax.annotate(f"{bf16[-1] - k4v4[-1]:.1f} MiB saved\nper sequence",
                xy=(30400, 495), xytext=(7600, 460), ha="center", va="center",
                fontsize=6.6, color="#8a5a00", bbox=BOX, arrowprops=dict(
                    arrowstyle="->", color=GOLD, linewidth=1.0,
                    connectionstyle="arc3,rad=-0.15"))
    ax.set_xticks(CONTEXTS, ["1K", "4K", "16K", "32K"])
    ax.set_yticks([8, 32, 128, 512, 1024], ["8", "32", "128", "512", "1024"])
    ax.minorticks_off()
    ax.set_xlabel("Context length", fontsize=8)
    ax.set_ylabel("Persistent draft KV (MiB)", fontsize=8)
    ax.set_title("(b) Packed savings scale with context", loc="left",
                 fontsize=9, fontweight="bold")
    style_axes(ax)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    width, height = 486.04, 197.48  # points, the size of the original figure
    fig = plt.figure(figsize=(width / 72, height / 72))
    axes =[fig.add_axes([x0 / width, 27.0 / height, (x1 - x0) / width, 119.4 / height])
            for x0, x1 in ((56.6, 241.7), (293.0, 450.0))]
    handles = operating_region(axes[0])
    packed_savings(axes[1])
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.003, 0.997),
               ncol=4, frameon=False, fontsize=6.5, handlelength=1.6,
               handletextpad=0.55, columnspacing=1.1, labelspacing=0.57,
               borderaxespad=0.2)
    fig.savefig(args.out, metadata={"CreationDate": None})
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
