#!/usr/bin/env python3
"""Regenerate preliminary paper figures and LaTeX tables from experiment JSON."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


PAPER_DIR = Path(__file__).resolve().parent
DATA_DIR = PAPER_DIR / "data" / "preliminary"
FIGURE_DIR = PAPER_DIR / "figures"
GENERATED_DIR = PAPER_DIR / "generated"

NAVY = "#17324D"
CORAL = "#D1495B"
TEAL = "#168C82"
GOLD = "#D99A2B"
INK = "#18212B"
GRID = "#D9DDD8"


def load_json(name: str):
    with (DATA_DIR / name).open("r", encoding="utf-8") as f:
        return json.load(f)


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "legend.frameon": False,
            "figure.dpi": 180,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )


def summary(path_name: str, config_name: str) -> dict:
    return load_json(path_name)["summaries"][config_name]


def plot_equal_memory_asymmetry() -> None:
    contexts = [
        ("1K", "qwen25_3b_15b_ctx1024.json"),
        ("4K", "qwen25_3b_15b_ctx4096.json"),
    ]
    configs = [("Native", "none", NAVY), ("K8 V4", "k8v4", TEAL), ("K4 V8", "k4v8", CORAL)]

    fig, ax = plt.subplots(figsize=(4.7, 2.45))
    width = 0.23
    x_positions = list(range(len(contexts)))
    for offset, (label, config_name, color) in enumerate(configs):
        values = [summary(path, config_name)["overall_accept_rate"] for _, path in contexts]
        bars = ax.bar(
            [x + (offset - 1) * width for x in x_positions],
            values,
            width=width,
            label=label,
            color=color,
        )
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.012, f"{value:.3f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x_positions, [f"{label} context" for label, _ in contexts])
    ax.set_ylabel("Speculative acceptance rate")
    ax.set_ylim(0, 0.68)
    ax.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    ax.set_title("Equal-memory K/V assignments are not equally safe", pad=18, fontweight="bold")
    fig.savefig(FIGURE_DIR / "equal_memory_asymmetry.pdf")
    fig.savefig(FIGURE_DIR / "equal_memory_asymmetry.png")
    plt.close(fig)


def plot_gaussian_sensitivity() -> None:
    rows = load_json("qwen25_15b_gaussian_all.json")
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.35))
    for target, label, color, marker in [
        ("keys", "Perturb K", CORAL, "o"),
        ("values", "Perturb V", TEAL, "s"),
    ]:
        selected = sorted(
            (row for row in rows if row["perturb_target"] == target and row["alpha"] > 0),
            key=lambda row: row["alpha"],
        )
        alpha = [row["alpha"] for row in selected]
        axes[0].plot(alpha, [row["top1_match"] for row in selected], label=label, color=color, marker=marker, markersize=3.5)
        axes[1].plot(alpha, [max(0.0, row["js"]) for row in selected], label=label, color=color, marker=marker, markersize=3.5)

    axes[0].set_title("Top-1 stability", fontweight="bold")
    axes[0].set_ylabel("Match to clean draft")
    axes[0].set_ylim(0, 1.05)
    axes[1].set_title("Distribution drift", fontweight="bold")
    axes[1].set_ylabel("Jensen-Shannon divergence")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel(r"Relative Gaussian noise $\alpha$")
        ax.grid(color=GRID, linewidth=0.7, alpha=0.8)
        ax.set_axisbelow(True)
    axes[0].legend(loc="lower left")
    fig.tight_layout(w_pad=1.8)
    fig.savefig(FIGURE_DIR / "gaussian_kv_sensitivity.pdf")
    fig.savefig(FIGURE_DIR / "gaussian_kv_sensitivity.png")
    plt.close(fig)


def plot_memory_acceptance_tradeoff() -> None:
    contexts = [
        ("1K", "qwen25_3b_15b_ctx1024.json", "o"),
        ("4K", "qwen25_3b_15b_ctx4096.json", "s"),
    ]
    colors = {"k8v8": GOLD, "k8v4": TEAL, "k4v8": CORAL, "k4v4": NAVY}
    labels = {"k8v8": "K8 V8", "k8v4": "K8 V4", "k4v8": "K4 V8", "k4v4": "K4 V4"}
    fig, ax = plt.subplots(figsize=(4.8, 2.6))

    for context_label, path, marker in contexts:
        baseline = summary(path, "none")["overall_accept_rate"]
        for config_name in ("k8v8", "k8v4", "k4v8", "k4v4"):
            row = summary(path, config_name)
            x = 100.0 * row["total_cache_saved_fraction"]
            y = 100.0 * row["overall_accept_rate"] / baseline
            ax.scatter(x, y, s=44, marker=marker, color=colors[config_name], edgecolor="white", linewidth=0.6, zorder=3)
            if context_label == "4K":
                ax.annotate(labels[config_name], (x, y), xytext=(4, 2), textcoords="offset points", fontsize=7.5)

    ax.axhline(100, color=INK, linewidth=0.8, linestyle="--", alpha=0.7)
    ax.set_xlabel("Total target + draft KV memory saved (%)")
    ax.set_ylabel("Acceptance retained (%)")
    ax.set_xlim(18, 35)
    ax.set_ylim(50, 104)
    ax.grid(color=GRID, linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Memory savings versus speculative acceptance", fontweight="bold")
    ax.text(0.02, 0.04, "circle: 1K   square: 4K", transform=ax.transAxes, fontsize=7.5)
    fig.savefig(FIGURE_DIR / "memory_acceptance_tradeoff.pdf")
    fig.savefig(FIGURE_DIR / "memory_acceptance_tradeoff.png")
    plt.close(fig)


def write_preliminary_table() -> None:
    rows = []
    for context, path in [("1K", "qwen25_3b_15b_ctx1024.json"), ("4K", "qwen25_3b_15b_ctx4096.json")]:
        baseline = summary(path, "none")["overall_accept_rate"]
        for config_name, display in [("none", "FP16"), ("k8v8", "K8 V8"), ("k8v4", "K8 V4"), ("k4v8", "K4 V8"), ("k4v4", "K4 V4")]:
            row = summary(path, config_name)
            rows.append(
                f"{context} & {display} & {row['overall_accept_rate']:.3f} & "
                f"{100.0 * row['overall_accept_rate'] / baseline:.1f} & "
                f"{100.0 * row['draft_cache_saved_fraction']:.1f} & "
                f"{100.0 * row['total_cache_saved_fraction']:.1f} & "
                f"{row['round_js']:.3f} \\\\"
            )

    table = "\n".join(
        [
            r"\begin{tabular}{llrrrrr}",
            r"\toprule",
            r"Context & Draft KV & Accept. & Retained (\%) & Draft saved (\%) & Total saved (\%) & JS $\downarrow$ \\",
            r"\midrule",
            *rows[:5],
            r"\midrule",
            *rows[5:],
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    (GENERATED_DIR / "preliminary_results_table.tex").write_text(table + "\n", encoding="utf-8")


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    configure_plot_style()
    plot_equal_memory_asymmetry()
    plot_gaussian_sensitivity()
    plot_memory_acceptance_tradeoff()
    write_preliminary_table()
    print(f"Wrote figures to {FIGURE_DIR}")
    print(f"Wrote tables to {GENERATED_DIR}")


if __name__ == "__main__":
    main()
