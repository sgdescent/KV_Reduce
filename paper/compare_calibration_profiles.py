#!/usr/bin/env python3
"""Compare speculative KV sensitivity profiles collected at different sample sizes."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRICS = (
    "accept_rate_drop_prompt_mean",
    "accept_rate_drop_ucb95_clipped",
    "accept_mass_drop_prompt_mean",
    "accept_mass_drop_ucb95_clipped",
)


def parse_profile(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Profiles must use LABEL=PATH syntax.")
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("Profiles must use non-empty LABEL=PATH syntax.")
    return label, Path(path)


def read_rows(path: Path) -> Dict[str, Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = {
            row["candidate"]: row
            for row in csv.DictReader(handle)
            if row.get("component") in {"k", "v"}
        }
    if not rows:
        raise ValueError(f"No K/V candidates found in {path}.")
    missing = [metric for metric in METRICS if metric not in next(iter(rows.values()))]
    if missing:
        raise ValueError(f"{path} is missing calibration statistics: {missing}")
    return rows


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else math.nan


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    mx, my = mean(xs), mean(ys)
    numerator = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denominator = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return numerator / denominator if denominator else math.nan


def ordinal_ranks(values: Sequence[float]) -> List[int]:
    ranks = [0] * len(values)
    for rank, index in enumerate(sorted(range(len(values)), key=values.__getitem__)):
        ranks[index] = rank
    return ranks


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    return pearson(ordinal_ranks(xs), ordinal_ranks(ys))


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def paired_stability(
    profiles: List[Tuple[str, Dict[str, Dict[str, str]]]], metric: str, top_k: int
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    reference_label, reference = profiles[-1]
    for label, candidate_rows in profiles[:-1]:
        common = sorted(set(candidate_rows) & set(reference))
        xs = [float(candidate_rows[name][metric]) for name in common]
        ys = [float(reference[name][metric]) for name in common]
        k = min(top_k, len(common))
        top_x = set(sorted(common, key=lambda name: float(candidate_rows[name][metric]), reverse=True)[:k])
        top_y = set(sorted(common, key=lambda name: float(reference[name][metric]), reverse=True)[:k])
        overlap = len(top_x & top_y)
        rows.append(
            {
                "profile": label,
                "reference": reference_label,
                "metric": metric,
                "num_common_candidates": len(common),
                "pearson": pearson(xs, ys),
                "spearman": spearman(xs, ys),
                "top_k": k,
                "top_k_overlap": overlap,
                "top_k_jaccard": overlap / len(top_x | top_y) if top_x or top_y else math.nan,
            }
        )
    return rows


def component_summary(
    profiles: List[Tuple[str, Dict[str, Dict[str, str]]]]
) -> List[Dict[str, object]]:
    output: List[Dict[str, object]] = []
    for label, rows in profiles:
        paired_counts = [float(row["paired_prompt_count"]) for row in rows.values() if row.get("paired_prompt_count")]
        for bits in sorted({int(float(row["bits"])) for row in rows.values()}):
            for component in ("k", "v"):
                selected = [
                    row
                    for row in rows.values()
                    if row["component"] == component and int(float(row["bits"])) == bits
                ]
                output.append(
                    {
                        "profile": label,
                        "paired_prompts": int(round(mean(paired_counts))),
                        "component": component,
                        "bits": bits,
                        **{metric: mean(float(row[metric]) for row in selected) for metric in METRICS},
                    }
                )
    return output


def plot_component_ucb(rows: List[Dict[str, object]], out_dir: Path) -> List[str]:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7))
    colors = {"k": "#c94b5f", "v": "#274466"}
    for axis, (metric, title) in zip(
        axes,
        [
            ("accept_rate_drop_ucb95_clipped", "Acceptance-rate UCB"),
            ("accept_mass_drop_ucb95_clipped", "Acceptance-mass UCB"),
        ],
    ):
        for component in ("k", "v"):
            selected = sorted(
                [row for row in rows if row["component"] == component and row["bits"] == 4],
                key=lambda row: int(row["paired_prompts"]),
            )
            axis.plot(
                [int(row["paired_prompts"]) for row in selected],
                [100.0 * float(row[metric]) for row in selected],
                marker="o",
                linewidth=2,
                color=colors[component],
                label=component.upper(),
            )
        axis.set_title(title)
        axis.set_xlabel("Calibration prompts")
        axis.set_ylabel("Mean upper-bound drop (percentage points)")
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    fig.suptitle("Sensitivity Estimates Stabilize with More Calibration Prompts", fontweight="bold")
    fig.tight_layout()
    paths = []
    for extension in ("png", "pdf"):
        path = out_dir / f"calibration_sample_size.{extension}"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        paths.append(str(path))
    plt.close(fig)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", action="append", type=parse_profile, required=True)
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()
    if len(args.profile) < 2:
        parser.error("Provide at least two --profile LABEL=PATH arguments.")

    profiles = [(label, read_rows(path)) for label, path in args.profile]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    component_rows = component_summary(profiles)
    stability_rows = []
    for metric in ("accept_rate_drop_ucb95_clipped", "accept_mass_drop_ucb95_clipped"):
        stability_rows.extend(paired_stability(profiles, metric, args.top_k))
    write_csv(args.out_dir / "component_sensitivity.csv", component_rows)
    write_csv(args.out_dir / "calibration_stability.csv", stability_rows)
    plots = plot_component_ucb(component_rows, args.out_dir)
    payload = {
        "profiles": [label for label, _ in profiles],
        "reference_profile": profiles[-1][0],
        "component_sensitivity": component_rows,
        "stability": stability_rows,
        "plots": plots,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Calibration comparison written to {args.out_dir}")


if __name__ == "__main__":
    main()
