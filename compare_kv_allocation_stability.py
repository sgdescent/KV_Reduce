#!/usr/bin/env python3
"""Compare mixed-precision KV allocations across calibration sample sizes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


def parse_allocation(value: str) -> Tuple[str, str, Path]:
    if "=" not in value or ":" not in value.split("=", 1)[0]:
        raise argparse.ArgumentTypeError(
            "Allocations must use OBJECTIVE:LABEL=PATH syntax."
        )
    descriptor, raw_path = value.split("=", 1)
    objective, label = descriptor.split(":", 1)
    if not objective or not label or not raw_path:
        raise argparse.ArgumentTypeError(
            "Allocations must use non-empty OBJECTIVE:LABEL=PATH fields."
        )
    return objective, label, Path(raw_path)


def read_bits(path: Path) -> Dict[str, List[int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = {
        "k": [int(value) for value in payload["k_bits"]],
        "v": [int(value) for value in payload["v_bits"]],
    }
    if len(result["k"]) != len(result["v"]):
        raise ValueError(f"K/V layer counts differ in {path}.")
    return result


def compare_bits(
    candidate: Dict[str, List[int]],
    reference: Dict[str, List[int]],
) -> Dict[str, float]:
    if len(candidate["k"]) != len(reference["k"]):
        raise ValueError("Allocation layer counts differ.")
    result: Dict[str, float] = {}
    all_candidate: List[int] = []
    all_reference: List[int] = []
    for component in ("k", "v"):
        first = candidate[component]
        second = reference[component]
        disagreements = sum(a != b for a, b in zip(first, second))
        absolute_difference = sum(abs(a - b) for a, b in zip(first, second))
        result[f"{component}_decision_count"] = float(len(first))
        result[f"{component}_disagreement_count"] = float(disagreements)
        result[f"{component}_disagreement_fraction"] = disagreements / max(1, len(first))
        result[f"{component}_mean_absolute_bit_difference"] = absolute_difference / max(
            1, len(first)
        )
        all_candidate.extend(first)
        all_reference.extend(second)
    total_disagreements = sum(a != b for a, b in zip(all_candidate, all_reference))
    total_absolute_difference = sum(
        abs(a - b) for a, b in zip(all_candidate, all_reference)
    )
    result.update(
        {
            "decision_count": float(len(all_candidate)),
            "disagreement_count": float(total_disagreements),
            "disagreement_fraction": total_disagreements / max(1, len(all_candidate)),
            "mean_absolute_bit_difference": total_absolute_difference
            / max(1, len(all_candidate)),
            "exact_allocation_match": float(total_disagreements == 0),
        }
    )
    return result


def stability_rows(
    allocations: Sequence[Tuple[str, str, Path]],
) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Tuple[str, Path]]] = {}
    for objective, label, path in allocations:
        grouped.setdefault(objective, []).append((label, path))
    rows: List[Dict[str, Any]] = []
    for objective, entries in grouped.items():
        if len(entries) < 2:
            raise ValueError(f"Objective {objective!r} needs at least two allocations.")
        reference_label, reference_path = entries[-1]
        reference = read_bits(reference_path)
        for label, path in entries[:-1]:
            rows.append(
                {
                    "objective": objective,
                    "profile": label,
                    "reference": reference_label,
                    **compare_bits(read_bits(path), reference),
                }
            )
    return rows


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allocation",
        action="append",
        type=parse_allocation,
        required=True,
        help="Repeated OBJECTIVE:LABEL=PATH allocation specification.",
    )
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()

    rows = stability_rows(args.allocation)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "allocation_stability.csv", rows)
    payload = {
        "reference_by_objective": {
            objective: label for objective, label, _ in args.allocation
        },
        "comparisons": rows,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(f"Allocation stability written to {args.out_dir}")


if __name__ == "__main__":
    main()
