#!/usr/bin/env python3
"""Plot V-MLP training summaries and eval metrics from a hybrid probe output directory."""
from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np


def load_csv(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def plot_v_mlp_by_layer(rows: List[Dict[str, Any]], out_path: str, title_suffix: str) -> None:
    L = [int(r["layer_idx"]) for r in rows]
    first_tr = [float(r["first_train_loss"]) for r in rows]
    last_tr = [float(r["last_train_loss"]) for r in rows]
    best_val = [float(r["best_val_loss"]) for r in rows]
    first_v = [float(r["first_val_loss"]) for r in rows]
    last_v = [float(r["last_val_loss"]) for r in rows]

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [1.2, 1]})

    ax = axes[0]
    ax.plot(L, first_tr, "o-", label="train first epoch", color="#c0392b", markersize=4)
    ax.plot(L, last_tr, "s-", label="train last epoch", color="#27ae60", markersize=4)
    ax.plot(L, first_v, "^--", label="val first epoch", color="#e67e22", alpha=0.85, markersize=4)
    ax.plot(L, best_val, "D-", label="val best (checkpoint)", color="#2980b9", markersize=4)
    ax.set_ylabel("loss (MSE + cos weighted)")
    ax.set_title(f"V-MLP loss by small-model layer {title_suffix}")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.35)
    ax.set_xlim(-0.5, max(L) + 0.5)

    d_tr = [float(r["train_loss_drop"]) for r in rows]
    d_va = [float(r["val_loss_drop"]) for r in rows]
    x = np.arange(len(L))
    w = 0.35
    axes[1].bar(x - w / 2, d_tr, width=w, label="train Δ (first − last)", color="#16a085", edgecolor="white", linewidth=0.5)
    axes[1].bar(x + w / 2, d_va, width=w, label="val Δ (first − last)", color="#8e44ad", edgecolor="white", linewidth=0.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([str(i) for i in L], rotation=45, ha="right", fontsize=7)
    axes[1].set_xlabel("layer_idx (small model)")
    axes[1].set_ylabel("loss drop")
    axes[1].set_title("Improvement over 5 epochs (larger = more learning)")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(True, axis="y", alpha=0.35)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_val_vs_train_final(rows: List[Dict[str, Any]], out_path: str, title_suffix: str) -> None:
    L = [int(r["layer_idx"]) for r in rows]
    last_tr = np.array([float(r["last_train_loss"]) for r in rows])
    best_val = np.array([float(r["best_val_loss"]) for r in rows])

    fig, ax = plt.subplots(figsize=(7, 7))
    sc = ax.scatter(last_tr, best_val, c=L, cmap="viridis", s=45, edgecolors="k", linewidths=0.3)
    lim = max(float(last_tr.max()), float(best_val.max())) * 1.05
    ax.plot([0, lim], [0, lim], "k--", alpha=0.4, label="y = x")
    ax.set_xlabel("final train loss")
    ax.set_ylabel("best val loss")
    ax.set_title(f"Train vs val (color = layer) {title_suffix}")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.35)
    ax.set_aspect("equal", adjustable="box")
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("layer_idx")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_summary_json(summary: dict, out_path: str) -> None:
    recon = summary.get("reconstruction_summary") or {}
    next_t = summary.get("next_token_summary") or {}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    if recon:
        keys = [k for k in sorted(recon.keys()) if k.endswith("_mse") or k.endswith("_cos")]
        vals = [float(recon[k]) for k in keys]
        colors = ["#c0392b" if "cos" in k else "#2980b9" for k in keys]
        axes[0].barh(keys, vals, color=colors, edgecolor="white")
        axes[0].set_xlabel("value (lower MSE / higher cos better)")
        axes[0].set_title("Reconstruction (mean over eval)")
        axes[0].grid(True, axis="x", alpha=0.35)

    match_keys = [k for k in sorted(next_t.keys()) if "top1_match" in k]
    if match_keys:
        mvals = [float(next_t[k]) for k in match_keys]
        axes[1].barh(match_keys, mvals, color="#16a085", edgecolor="white")
        axes[1].set_xlim(0, 1)
        axes[1].set_xlabel("rate")
        axes[1].set_title("Next-token top-1 match vs reference")
        axes[1].grid(True, axis="x", alpha=0.35)

    fig.suptitle("summary.json aggregates", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str, default="outputs/qwen25_3b_to_15b_hybrid_vmlp_v1")
    args = p.parse_args()
    run_dir = args.run_dir
    csv_path = os.path.join(run_dir, "v_mlp_train_summary.csv")
    json_path = os.path.join(run_dir, "summary.json")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(csv_path)

    rows = load_csv(csv_path)
    title_suffix = f"({os.path.basename(run_dir.rstrip('/'))})"

    os.makedirs(run_dir, exist_ok=True)
    plot_v_mlp_by_layer(rows, os.path.join(run_dir, "figures_v_mlp_by_layer.png"), title_suffix)
    plot_val_vs_train_final(rows, os.path.join(run_dir, "figures_v_mlp_train_vs_val.png"), title_suffix)
    if os.path.isfile(json_path):
        plot_summary_json(load_json(json_path), os.path.join(run_dir, "figures_summary_metrics.png"))

    print("Wrote:")
    print(f"  {os.path.join(run_dir, 'figures_v_mlp_by_layer.png')}")
    print(f"  {os.path.join(run_dir, 'figures_v_mlp_train_vs_val.png')}")
    if os.path.isfile(json_path):
        print(f"  {os.path.join(run_dir, 'figures_summary_metrics.png')}")
    print("\nNote: Per-epoch curves were not saved for this run; only per-layer aggregates from v_mlp_train_summary.csv.")


if __name__ == "__main__":
    main()
