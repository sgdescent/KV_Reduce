#!/usr/bin/env python3
"""Build a matched-memory family of quality-prior shrinkage allocations."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


def parse_floats(value: str) -> List[float]:
    return [float(item.strip()) for item in value.replace(";", ",").split(",") if item.strip()]


def strength_label(value: float) -> str:
    return f"p{value:g}".replace(".", "p")


def allocation_name(path: Path) -> str:
    return str(json.loads(path.read_text(encoding="utf-8"))["name"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality_profile_csv", required=True)
    parser.add_argument("--acceptance_profile_csv", required=True)
    parser.add_argument("--quality_allocation", required=True)
    parser.add_argument("--acceptance_allocation", required=True)
    parser.add_argument("--num_layers", type=int, required=True)
    parser.add_argument("--target_profiled_mean_bits", type=float, default=8.0)
    parser.add_argument("--prior_strengths", default="0.25,1,4")
    parser.add_argument("--include_ucb", action="store_true")
    parser.add_argument("--out_dir", required=True)
    return parser


def run(command: List[str]) -> None:
    subprocess.run(command, check=True)


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.out_dir)
    root.mkdir(parents=True, exist_ok=True)
    quality_allocation = Path(args.quality_allocation)
    acceptance_allocation = Path(args.acceptance_allocation)
    policies: List[Dict[str, str]] = [
        {
            "kind": "native",
            "name": "none",
            "config": "none",
        },
        {
            "kind": "uniform",
            "name": "k8v8",
            "config": "k8v8",
        },
        {
            "kind": "quality",
            "name": allocation_name(quality_allocation),
            "config": f"allocation:{quality_allocation}",
        },
        {
            "kind": "raw_acceptance",
            "name": allocation_name(acceptance_allocation),
            "config": f"allocation:{acceptance_allocation}",
        },
    ]

    specifications = [(strength, 0.0) for strength in parse_floats(args.prior_strengths)]
    if args.include_ucb:
        specifications.append((1.0, 1.96))
    for strength, ucb_z in specifications:
        label = strength_label(strength) + ("_ucb95" if ucb_z else "")
        profile_dir = root / "profiles" / label
        allocation_dir = root / "allocations" / label
        risk_field = "shrunk_acceptance_ucb" if ucb_z else "shrunk_acceptance_risk"
        run(
            [
                sys.executable,
                "build_shrunk_kv_profile.py",
                "--quality_profile_csv",
                args.quality_profile_csv,
                "--acceptance_profile_csv",
                args.acceptance_profile_csv,
                "--prior_strength",
                str(strength),
                "--ucb_z",
                str(ucb_z),
                "--variance_mode",
                "empirical",
                "--out_dir",
                str(profile_dir),
            ]
        )
        name = f"shrunk_{label}"
        run(
            [
                sys.executable,
                "search_kv_bit_allocation.py",
                "--profile_csv",
                str(profile_dir / "profile_summary.csv"),
                "--num_layers",
                str(args.num_layers),
                "--risk_field",
                risk_field,
                "--target_profiled_mean_bits",
                str(args.target_profiled_mean_bits),
                "--name",
                name,
                "--out_dir",
                str(allocation_dir),
            ]
        )
        policies.append(
            {
                "kind": "shrunk_acceptance",
                "name": name,
                "config": f"allocation:{allocation_dir / 'allocation.json'}",
                "prior_strength": str(strength),
                "ucb_z": str(ucb_z),
                "risk_field": risk_field,
            }
        )

    bundle = {
        "runtime": {"allocator_version": "quality_prior_shrinkage_v1"},
        "target_profiled_mean_bits": args.target_profiled_mean_bits,
        "policies": policies,
        "quant_configs": ";".join(policy["config"] for policy in policies),
    }
    (root / "bundle.json").write_text(
        json.dumps(bundle, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(bundle, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
