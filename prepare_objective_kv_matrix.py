#!/usr/bin/env python3
"""Prepare equal-memory allocations and a bounded cross-evaluation manifest."""

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path
from typing import List


def parse_csv_ints(value: str) -> List[int]:
    return [int(item.strip()) for item in value.replace(";", ",").split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare the objective-aware KV evaluation matrix.")
    parser.add_argument("--quality_profile_csv", required=True)
    parser.add_argument("--acceptance_profile_csv", required=True)
    parser.add_argument("--num_layers", type=int, required=True)
    parser.add_argument("--budgets", default="6,8,10,12")
    parser.add_argument("--contexts", default="512,1024,4096")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--num_eval", type=int, default=32)
    parser.add_argument("--out_dir", required=True)
    return parser


def run_allocator(
    *,
    profile_csv: str,
    risk_field: str,
    budget: int,
    name: str,
    out_dir: Path,
    num_layers: int,
) -> Path:
    subprocess.run(
        [
            sys.executable,
            "search_kv_bit_allocation.py",
            "--profile_csv",
            profile_csv,
            "--num_layers",
            str(num_layers),
            "--risk_field",
            risk_field,
            "--target_profiled_mean_bits",
            str(budget),
            "--name",
            name,
            "--out_dir",
            str(out_dir),
        ],
        check=True,
    )
    return out_dir / "allocation.json"


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.out_dir)
    root.mkdir(parents=True, exist_ok=True)
    budgets = parse_csv_ints(args.budgets)
    contexts = parse_csv_ints(args.contexts)
    seeds = parse_csv_ints(args.seeds)
    rows = []

    for budget in budgets:
        budget_root = root / f"budget_{budget}"
        quality_allocation = run_allocator(
            profile_csv=args.quality_profile_csv,
            risk_field="quality_risk",
            budget=budget,
            name=f"quality_b{budget}",
            out_dir=budget_root / "quality_allocation",
            num_layers=args.num_layers,
        )
        acceptance_allocation = run_allocator(
            profile_csv=args.acceptance_profile_csv,
            risk_field="accept_rate_drop",
            budget=budget,
            name=f"acceptance_b{budget}",
            out_dir=budget_root / "acceptance_allocation",
            num_layers=args.num_layers,
        )
        configs = f"none;allocation:{quality_allocation};allocation:{acceptance_allocation}"
        for context in contexts:
            for seed in seeds:
                eval_root = budget_root / f"ctx_{context}" / f"seed_{seed}"
                for objective in ("quality", "acceptance"):
                    rows.append(
                        {
                            "objective": objective,
                            "budget": budget,
                            "context": context,
                            "seed": seed,
                            "num_eval": args.num_eval,
                            "quant_configs": configs,
                            "quality_allocation": str(quality_allocation),
                            "acceptance_allocation": str(acceptance_allocation),
                            "out_dir": str(eval_root / objective),
                        }
                    )

    manifest_path = root / "manifest.tsv"
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Prepared {len(rows)} evaluation tasks")
    print(manifest_path)


if __name__ == "__main__":
    main()
