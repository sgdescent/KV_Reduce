#!/usr/bin/env python3
"""Prepare equal-memory allocations and a bounded cross-evaluation manifest."""

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List


FULL_PRECISION_BITS = 16


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
    parser.add_argument("--quality_risk_field", default="quality_risk")
    parser.add_argument("--acceptance_risk_field", default="accept_rate_drop")
    parser.add_argument("--out_dir", required=True)
    return parser


def heuristic_component_bits(budget: int, *, prioritize: str) -> tuple[int, int]:
    """Return an exact-mean K/V heuristic using the supported 4/8/16-bit levels."""
    if prioritize not in {"k", "v"}:
        raise ValueError("prioritize must be 'k' or 'v'.")
    supported = {
        # At the minimum supported budget there is no asymmetric split, but the
        # uniform 4-bit endpoint is still needed to complete the memory/quality
        # Pareto curve for all-layer experiments.
        4: (4, 4),
        6: (8, 4),
        8: (8, 8),
        10: (16, 4),
        12: (16, 8),
        16: (16, 16),
    }
    if budget not in supported:
        raise ValueError(f"No exact K/V heuristic is defined for a {budget}-bit mean budget.")
    k_bits, v_bits = supported[budget]
    return (k_bits, v_bits) if prioritize == "k" else (v_bits, k_bits)


def read_profiled_layers(profile_csv: str) -> List[int]:
    with open(profile_csv, "r", encoding="utf-8", newline="") as f:
        rows = csv.DictReader(f)
        layers = {
            int(float(row["layer"]))
            for row in rows
            if row.get("component") in {"k", "v"} and int(float(row.get("layer", -1))) >= 0
        }
    if not layers:
        raise ValueError(f"No profiled K/V layers were found in {profile_csv}.")
    return sorted(layers)


def write_heuristic_allocation(
    *,
    budget: int,
    prioritize: str,
    profiled_layers: List[int],
    num_layers: int,
    out_dir: Path,
) -> Path:
    profiled_k_bits, profiled_v_bits = heuristic_component_bits(budget, prioritize=prioritize)
    k_bits = [FULL_PRECISION_BITS] * num_layers
    v_bits = [FULL_PRECISION_BITS] * num_layers
    for layer in profiled_layers:
        k_bits[layer] = profiled_k_bits
        v_bits[layer] = profiled_v_bits
    name = f"{prioritize}_priority_b{budget}"
    payload = {
        "name": name,
        "source": "matched_memory_hand_designed_baseline",
        "priority": prioritize,
        "target_profiled_mean_bits": budget,
        "achieved_profiled_mean_bits": (profiled_k_bits + profiled_v_bits) / 2.0,
        "k_bits": k_bits,
        "v_bits": v_bits,
        "layers": [
            {"layer": layer, "k_bits": k_bits[layer], "v_bits": v_bits[layer]}
            for layer in range(num_layers)
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "allocation.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


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
    profiled_layers = read_profiled_layers(args.acceptance_profile_csv)

    for budget in budgets:
        budget_root = root / f"budget_{budget}"
        quality_allocation = run_allocator(
            profile_csv=args.quality_profile_csv,
            risk_field=args.quality_risk_field,
            budget=budget,
            name=f"quality_b{budget}",
            out_dir=budget_root / "quality_allocation",
            num_layers=args.num_layers,
        )
        acceptance_allocation = run_allocator(
            profile_csv=args.acceptance_profile_csv,
            risk_field=args.acceptance_risk_field,
            budget=budget,
            name=f"acceptance_b{budget}",
            out_dir=budget_root / "acceptance_allocation",
            num_layers=args.num_layers,
        )
        k_priority_allocation = write_heuristic_allocation(
            budget=budget,
            prioritize="k",
            profiled_layers=profiled_layers,
            num_layers=args.num_layers,
            out_dir=budget_root / "k_priority_allocation",
        )
        v_priority_allocation = write_heuristic_allocation(
            budget=budget,
            prioritize="v",
            profiled_layers=profiled_layers,
            num_layers=args.num_layers,
            out_dir=budget_root / "v_priority_allocation",
        )
        configs = (
            f"none;allocation:{quality_allocation};allocation:{acceptance_allocation};"
            f"allocation:{k_priority_allocation};allocation:{v_priority_allocation}"
        )
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
                            "k_priority_allocation": str(k_priority_allocation),
                            "v_priority_allocation": str(v_priority_allocation),
                            "out_dir": str(eval_root / objective),
                        }
                    )

    manifest_path = root / "manifest.tsv"
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        # Bash reads the manifest line-by-line; avoid CSV's default CRLF so the
        # final out_dir field does not acquire a literal carriage return.
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Prepared {len(rows)} evaluation tasks")
    print(manifest_path)


if __name__ == "__main__":
    main()
