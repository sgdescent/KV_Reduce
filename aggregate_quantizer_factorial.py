#!/usr/bin/env python3
"""Aggregate the 2x2 key-axis/value-scheme quantizer factorial."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


CELL_NAMES = {
    "per_token_symmetric",
    "per_token_affine",
    "per_channel_symmetric",
    "per_channel_affine",
}


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
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


def parse_cell(spec: str) -> Tuple[str, Path, Path, str | None]:
    fields = [field.strip() for field in spec.split(",")]
    if len(fields) not in {3, 4}:
        raise ValueError("Each --cell must be NAME,SPEC_SUMMARY,QUALITY_SUMMARY[,SPEC_VARIANT].")
    name = fields[0]
    if name not in CELL_NAMES:
        raise ValueError(f"Unsupported factorial cell {name!r}.")
    return name, Path(fields[1]), Path(fields[2]), fields[3] if len(fields) == 4 else None


def acceptance_metric(row: Dict[str, Any]) -> Tuple[float, float, float]:
    if "paired_acceptance_delta_mean" in row:
        return (
            float(row["paired_acceptance_delta_mean"]),
            float(row["paired_acceptance_delta_ci_low"]),
            float(row["paired_acceptance_delta_ci_high"]),
        )
    return (
        float(row["variant_effect_mean"]),
        float(row["variant_effect_ci_low"]),
        float(row["variant_effect_ci_high"]),
    )


def load_cell(
    name: str,
    spec_summary: Path,
    quality_summary: Path,
    spec_variant: str | None,
) -> Dict[Tuple[int, str], Dict[str, Any]]:
    spec_rows = {}
    for row in read_json(spec_summary)["grouped"]:
        if spec_variant is not None and row.get("variant") != spec_variant:
            continue
        key = (int(row["context"]), str(row["config"]))
        if key in spec_rows:
            raise ValueError(f"Duplicate speculative row for {name} {key}; provide a variant filter.")
        spec_rows[key] = row
    quality_rows = {
        (int(row["context"]), str(row["config"])): row
        for row in read_json(quality_summary)["grouped"]
    }
    output = {}
    for key in sorted(spec_rows.keys() & quality_rows.keys()):
        spec_row = spec_rows[key]
        quality_row = quality_rows[key]
        acceptance, acceptance_low, acceptance_high = acceptance_metric(spec_row)
        output[key] = {
            "cell": name,
            "context": key[0],
            "config": key[1],
            "acceptance_delta": acceptance,
            "acceptance_delta_ci_low": acceptance_low,
            "acceptance_delta_ci_high": acceptance_high,
            "quality_kl": float(quality_row["kl_p_to_q_mean"]),
            "quality_kl_ci_low": float(quality_row["kl_p_to_q_ci_low"]),
            "quality_kl_ci_high": float(quality_row["kl_p_to_q_ci_high"]),
            "total_cache_saved_fraction": float(spec_row["total_cache_saved_fraction"]),
        }
    if not output:
        raise ValueError(f"No joined speculative/quality rows for cell {name}.")
    return output


def factorial_effects(cells: Dict[str, Dict[Tuple[int, str], Dict[str, Any]]]) -> List[Dict[str, Any]]:
    if set(cells) != CELL_NAMES:
        raise ValueError(f"Expected cells {sorted(CELL_NAMES)}, received {sorted(cells)}.")
    common = set.intersection(*(set(rows) for rows in cells.values()))
    output = []
    for context, config in sorted(common):
        pt_sym = cells["per_token_symmetric"][(context, config)]
        pt_aff = cells["per_token_affine"][(context, config)]
        pc_sym = cells["per_channel_symmetric"][(context, config)]
        pc_aff = cells["per_channel_affine"][(context, config)]
        row: Dict[str, Any] = {"context": context, "config": config}
        for metric in ("acceptance_delta", "quality_kl"):
            row[f"key_axis_effect_under_symmetric/{metric}"] = pc_sym[metric] - pt_sym[metric]
            row[f"key_axis_effect_under_affine/{metric}"] = pc_aff[metric] - pt_aff[metric]
            row[f"value_affine_effect_under_per_token/{metric}"] = pt_aff[metric] - pt_sym[metric]
            row[f"value_affine_effect_under_per_channel/{metric}"] = pc_aff[metric] - pc_sym[metric]
            row[f"interaction/{metric}"] = (pc_aff[metric] - pt_aff[metric]) - (
                pc_sym[metric] - pt_sym[metric]
            )
        output.append(row)
    return output


def summarize_effects(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["context"])].append(row)
    output = []
    for context, context_rows in sorted(grouped.items()):
        summary: Dict[str, Any] = {"context": context, "num_configs": len(context_rows)}
        for key in context_rows[0]:
            if key in {"context", "config"}:
                continue
            summary[f"mean/{key}"] = statistics.mean(float(row[key]) for row in context_rows)
        output.append(summary)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", action="append", required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cells = {}
    cell_rows = []
    for raw_cell in args.cell:
        name, spec_path, quality_path, variant = parse_cell(raw_cell)
        if name in cells:
            raise ValueError(f"Duplicate factorial cell {name!r}.")
        cells[name] = load_cell(name, spec_path, quality_path, variant)
        cell_rows.extend(cells[name].values())

    effects = factorial_effects(cells)
    summaries = summarize_effects(effects)
    write_csv(args.out_dir / "cell_metrics.csv", cell_rows)
    write_csv(args.out_dir / "factorial_effects.csv", effects)
    write_csv(args.out_dir / "factorial_summary.csv", summaries)
    payload = {
        "num_cells": len(cells),
        "num_complete_factorial_rows": len(effects),
        "cells": sorted(cells),
        "contexts": summaries,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(args.out_dir / "summary.json")


if __name__ == "__main__":
    main()
