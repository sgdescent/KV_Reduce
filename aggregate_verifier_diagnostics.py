#!/usr/bin/env python3
"""Aggregate cached target-verification numerical diagnostics."""

from __future__ import annotations

import argparse
import csv
import glob
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


COUNT_FIELDS = (
    "batch_vs_sequential_top1_mismatches",
    "full_vs_sequential_top1_mismatches",
    "causal_suffix_violations",
    "rollback_top1_mismatches",
    "speculative_top1_mismatches",
    "speculative_independent_greedy_mismatches",
)
MAX_FIELDS = (
    "max_batch_vs_sequential_logit_delta",
    "max_causal_suffix_logit_delta",
    "max_rollback_logit_delta",
)


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, help="JSON paths or glob patterns.")
    parser.add_argument("--out_dir", required=True, type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    paths = sorted(
        {
            Path(match)
            for pattern in args.inputs
            for match in glob.glob(pattern)
        }
    )
    if not paths:
        raise ValueError("No verifier diagnostic JSON files matched --inputs.")

    rows: List[Dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        config = payload["config"]
        row = {
            "path": str(path),
            "dtype": config["dtype"],
            "attn_implementation": config["attn_implementation"],
            "seed": config["seed"],
            "skip_prompts": config["skip_prompts"],
            "num_prompts": int(payload.get("num_prompts", 1)),
            "prompts_with_speculative_mismatch": int(
                payload.get(
                    "speculative_prompts_with_top1_mismatch",
                    int(payload.get("speculative_top1_mismatches", 0)) > 0,
                )
            ),
            "prompts_with_independent_greedy_mismatch": int(
                payload.get(
                    "speculative_prompts_with_independent_greedy_mismatch",
                    int(payload.get("speculative_independent_greedy_mismatches", 0)) > 0,
                )
            ),
        }
        row.update({field: int(payload[field]) for field in COUNT_FIELDS})
        row.update({field: float(payload[field]) for field in MAX_FIELDS})
        rows.append(row)

    grouped_rows: List[Dict[str, Any]] = []
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["dtype"]), str(row["attn_implementation"]))].append(row)
    for (dtype, backend), values in sorted(grouped.items()):
        grouped_row: Dict[str, Any] = {
            "dtype": dtype,
            "attn_implementation": backend,
            "num_runs": len(values),
            "num_prompts": sum(int(row["num_prompts"]) for row in values),
            "prompts_with_speculative_mismatch": sum(
                int(row["prompts_with_speculative_mismatch"]) for row in values
            ),
            "prompts_with_independent_greedy_mismatch": sum(
                int(row["prompts_with_independent_greedy_mismatch"])
                for row in values
            ),
        }
        grouped_row.update(
            {f"total_{field}": sum(int(row[field]) for row in values) for field in COUNT_FIELDS}
        )
        grouped_row.update(
            {field: max(float(row[field]) for row in values) for field in MAX_FIELDS}
        )
        grouped_rows.append(grouped_row)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "runs.csv", rows)
    write_csv(args.out_dir / "grouped.csv", grouped_rows)
    summary = {
        "num_runs": len(rows),
        "num_unique_prompts": len({(row["seed"], row["skip_prompts"]) for row in rows}),
        "grouped": grouped_rows,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(args.out_dir / "summary.json")


if __name__ == "__main__":
    main()
