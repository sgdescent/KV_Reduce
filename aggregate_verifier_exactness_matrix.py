#!/usr/bin/env python3
"""Aggregate backend/reset verifier diagnostics with an explicit tie audit."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


EVALUATOR_VERSION = "verifier_exactness_matrix_v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Directory containing diagnostic JSON files.")
    parser.add_argument("--tie_margin", type=float, default=1e-3)
    parser.add_argument("--out_dir", required=True)
    return parser


def iter_decisions(payload: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for audit in payload.get("speculative_audits", []):
        yield from audit.get("decisions", [])


def summarize(path: Path, tie_margin: float) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    config = payload.get("config", {})
    decisions = list(iter_decisions(payload))
    mismatches = [row for row in decisions if float(row.get("top1_match", 1.0)) < 0.5]
    tie_mismatches = []
    non_tie_mismatches = []
    for row in mismatches:
        margins = [
            float(row.get("reference_margin", float("inf"))),
            float(row.get("candidate_margin", float("inf"))),
        ]
        if min(margins) <= tie_margin:
            tie_mismatches.append(row)
        else:
            non_tie_mismatches.append(row)

    return {
        "file": path.name,
        "model": config.get("model"),
        "small_model": config.get("small_model"),
        "dtype": config.get("dtype"),
        "attn_implementation": config.get("attn_implementation"),
        "reset_target_from_sequential_shadow": bool(
            config.get("reset_target_from_sequential_shadow", False)
        ),
        "stream_eval": bool(config.get("stream_eval", False)),
        "num_prompts": int(payload.get("num_speculative_audit_prompts", 0)),
        "num_decisions": len(decisions),
        "top1_mismatches": len(mismatches),
        "numerical_tie_mismatches": len(tie_mismatches),
        "non_tie_mismatches": len(non_tie_mismatches),
        "prompts_with_top1_mismatch": int(
            payload.get("speculative_prompts_with_top1_mismatch", 0)
        ),
        "prompts_with_independent_greedy_mismatch": int(
            payload.get("speculative_prompts_with_independent_greedy_mismatch", 0)
        ),
        "max_abs_logit_delta": max(
            (float(row.get("max_abs_logit_delta", 0.0)) for row in decisions),
            default=0.0,
        ),
    }


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.root)
    paths = sorted(path for path in root.glob("*.json") if path.name != "summary.json")
    if not paths:
        raise FileNotFoundError(f"No diagnostic JSON files found under {root}")

    rows: List[Dict[str, Any]] = [summarize(path, args.tie_margin) for path in paths]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "matrix.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "runtime": {
            "evaluator_version": EVALUATOR_VERSION,
            "tie_margin": args.tie_margin,
        },
        "num_cells": len(rows),
        "rows": rows,
        "any_non_tie_mismatch": any(row["non_tie_mismatches"] > 0 for row in rows),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
