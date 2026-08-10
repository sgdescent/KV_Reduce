#!/usr/bin/env python3
"""Prepare a speculative draft-length ablation from existing allocations."""

import argparse
import csv
from pathlib import Path
from typing import List


def parse_csv_ints(value: str) -> List[int]:
    return [int(item.strip()) for item in value.replace(";", ",").split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare the speculative KV draft-length matrix.")
    parser.add_argument("--allocation_root", required=True)
    parser.add_argument("--budgets", default="6,8")
    parser.add_argument("--contexts", default="1024,4096")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--draft_steps", default="2,4,8")
    parser.add_argument("--num_eval", type=int, default=32)
    parser.add_argument("--out_dir", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    allocation_root = Path(args.allocation_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for budget in parse_csv_ints(args.budgets):
        budget_root = allocation_root / f"budget_{budget}"
        allocation_paths = [
            budget_root / "quality_allocation" / "allocation.json",
            budget_root / "acceptance_allocation" / "allocation.json",
            budget_root / "k_priority_allocation" / "allocation.json",
            budget_root / "v_priority_allocation" / "allocation.json",
        ]
        missing = [str(path) for path in allocation_paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing allocation files: {missing}")
        quant_configs = "none;" + ";".join(f"allocation:{path}" for path in allocation_paths)
        for context in parse_csv_ints(args.contexts):
            for seed in parse_csv_ints(args.seeds):
                for draft_steps in parse_csv_ints(args.draft_steps):
                    rows.append(
                        {
                            "budget": budget,
                            "context": context,
                            "seed": seed,
                            "draft_steps": draft_steps,
                            "num_eval": args.num_eval,
                            "quant_configs": quant_configs,
                            "out_dir": str(
                                out_dir
                                / f"budget_{budget}"
                                / f"ctx_{context}"
                                / f"gamma_{draft_steps}"
                                / f"seed_{seed}"
                            ),
                        }
                    )
    manifest = out_dir / "manifest.tsv"
    with manifest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Prepared {len(rows)} gamma-ablation tasks")
    print(manifest)


if __name__ == "__main__":
    main()
