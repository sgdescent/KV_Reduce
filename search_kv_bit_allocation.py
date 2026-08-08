#!/usr/bin/env python3
"""
Build a mixed-precision draft KV allocation from sensitivity profile results.

This is intentionally a lightweight, transparent search. It treats one-at-a-time
profile drops as an additive risk proxy and chooses the lowest-bit candidate for
each layer/component that stays within configurable per-component and total
acceptance-drop budgets.
"""

import argparse
import csv
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Tuple

from kv_cache_quantization import FULL_PRECISION_BITS, parse_csv_ints, uniform_bit_lists
from kv_utils import write_json


def read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows: List[Dict[str, Any]], path: str) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def as_float(row: Dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    if value is None or value == "":
        return default
    return float(value)


def as_int(row: Dict[str, str], key: str, default: int = 0) -> int:
    value = row.get(key)
    if value is None or value == "":
        return default
    return int(float(value))


def candidate_saved_bytes(row: Dict[str, str]) -> float:
    for key in ("total_cache_mib_saved", "cache_mib_saved"):
        if key in row and row[key] != "":
            return as_float(row, key) * (1024.0**2)
    native = as_float(row, "native_total_cache_bytes", 0.0)
    quantized = as_float(row, "quantized_total_cache_bytes", native)
    return max(0.0, native - quantized)


def choose_initial_candidate(
    candidates: List[Dict[str, str]],
    *,
    allowed_bits: List[int],
    max_component_drop: float,
) -> Dict[str, Any]:
    viable = []
    for row in candidates:
        bits = as_int(row, "bits", FULL_PRECISION_BITS)
        if bits not in allowed_bits:
            continue
        drop = max(0.0, as_float(row, "accept_rate_drop", 0.0))
        if drop <= max_component_drop:
            viable.append(row)
    if not viable:
        return {
            "bits": FULL_PRECISION_BITS,
            "accept_rate_drop": 0.0,
            "saved_bytes": 0.0,
            "source_candidate": "full_precision_fallback",
        }
    viable.sort(key=lambda row: (as_int(row, "bits", FULL_PRECISION_BITS), as_float(row, "accept_rate_drop", 0.0)))
    chosen = viable[0]
    return {
        "bits": as_int(chosen, "bits", FULL_PRECISION_BITS),
        "accept_rate_drop": max(0.0, as_float(chosen, "accept_rate_drop", 0.0)),
        "saved_bytes": candidate_saved_bytes(chosen),
        "source_candidate": chosen.get("candidate", ""),
    }


def safer_replacement(
    candidates: List[Dict[str, str]],
    *,
    current_bits: int,
    allowed_bits: List[int],
) -> Dict[str, Any]:
    safer = []
    for row in candidates:
        bits = as_int(row, "bits", FULL_PRECISION_BITS)
        if bits not in allowed_bits:
            continue
        if bits <= current_bits:
            continue
        safer.append(row)
    if not safer:
        return {
            "bits": FULL_PRECISION_BITS,
            "accept_rate_drop": 0.0,
            "saved_bytes": 0.0,
            "source_candidate": "full_precision_relax",
        }
    safer.sort(key=lambda row: (as_int(row, "bits", FULL_PRECISION_BITS), as_float(row, "accept_rate_drop", 0.0)))
    chosen = safer[0]
    return {
        "bits": as_int(chosen, "bits", FULL_PRECISION_BITS),
        "accept_rate_drop": max(0.0, as_float(chosen, "accept_rate_drop", 0.0)),
        "saved_bytes": candidate_saved_bytes(chosen),
        "source_candidate": chosen.get("candidate", ""),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Search a mixed-precision KV allocation from sensitivity results.")
    parser.add_argument("--profile_csv", type=str, required=True)
    parser.add_argument("--num_layers", type=int, default=None)
    parser.add_argument("--allowed_bits", type=str, default="4,8,16")
    parser.add_argument("--max_component_drop", type=float, default=0.01)
    parser.add_argument("--max_total_drop", type=float, default=0.05)
    parser.add_argument("--name", type=str, default="sensitivity_aware")
    parser.add_argument("--out_dir", type=str, default="outputs/spec_kv_allocation")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rows = read_csv(args.profile_csv)
    allowed_bits = sorted(set(parse_csv_ints(args.allowed_bits)))
    if FULL_PRECISION_BITS not in allowed_bits:
        allowed_bits.append(FULL_PRECISION_BITS)
        allowed_bits = sorted(set(allowed_bits))

    profile_rows = [
        row
        for row in rows
        if row.get("component") in {"k", "v"} and as_int(row, "layer", -1) >= 0
    ]
    if not profile_rows:
        raise ValueError("No layer/component rows found in profile CSV.")

    num_layers = args.num_layers
    if num_layers is None:
        num_layers = max(as_int(row, "layer", 0) for row in profile_rows) + 1

    grouped: Dict[Tuple[int, str], List[Dict[str, str]]] = defaultdict(list)
    for row in profile_rows:
        grouped[(as_int(row, "layer"), str(row["component"]))].append(row)

    k_bits, v_bits = uniform_bit_lists(num_layers, FULL_PRECISION_BITS, FULL_PRECISION_BITS)
    selections: Dict[Tuple[int, str], Dict[str, Any]] = {}
    for layer in range(num_layers):
        for component in ("k", "v"):
            candidates = grouped.get((layer, component), [])
            if not candidates:
                selections[(layer, component)] = {
                    "bits": FULL_PRECISION_BITS,
                    "accept_rate_drop": 0.0,
                    "saved_bytes": 0.0,
                    "source_candidate": "unprofiled_full_precision",
                }
                continue
            selections[(layer, component)] = choose_initial_candidate(
                candidates,
                allowed_bits=allowed_bits,
                max_component_drop=args.max_component_drop,
            )

    def estimated_drop() -> float:
        return float(sum(selection["accept_rate_drop"] for selection in selections.values()))

    while estimated_drop() > args.max_total_drop:
        compressive = [
            (key, selection)
            for key, selection in selections.items()
            if int(selection["bits"]) < FULL_PRECISION_BITS and selection["saved_bytes"] > 0
        ]
        if not compressive:
            break
        compressive.sort(
            key=lambda item: (
                item[1]["accept_rate_drop"] / max(item[1]["saved_bytes"], 1.0),
                item[1]["accept_rate_drop"],
            ),
            reverse=True,
        )
        key, selection = compressive[0]
        replacement = safer_replacement(
            grouped.get(key, []),
            current_bits=int(selection["bits"]),
            allowed_bits=allowed_bits,
        )
        if int(replacement["bits"]) == int(selection["bits"]):
            break
        selections[key] = replacement

    selected_rows: List[Dict[str, Any]] = []
    for layer in range(num_layers):
        for component in ("k", "v"):
            selection = selections[(layer, component)]
            if component == "k":
                k_bits[layer] = int(selection["bits"])
            else:
                v_bits[layer] = int(selection["bits"])
            selected_rows.append(
                {
                    "layer": layer,
                    "component": component,
                    "bits": int(selection["bits"]),
                    "accept_rate_drop": float(selection["accept_rate_drop"]),
                    "saved_bytes": float(selection["saved_bytes"]),
                    "saved_mib": float(selection["saved_bytes"]) / (1024.0**2),
                    "source_candidate": selection["source_candidate"],
                }
            )

    payload = {
        "name": args.name,
        "source_profile_csv": args.profile_csv,
        "allowed_bits": allowed_bits,
        "max_component_drop": args.max_component_drop,
        "max_total_drop": args.max_total_drop,
        "estimated_accept_rate_drop": estimated_drop(),
        "estimated_saved_bytes_proxy": float(sum(selection["saved_bytes"] for selection in selections.values())),
        "k_bits": k_bits,
        "v_bits": v_bits,
        "layers": [
            {"layer": layer, "k_bits": int(k_bits[layer]), "v_bits": int(v_bits[layer])}
            for layer in range(num_layers)
        ],
    }
    write_json(payload, os.path.join(args.out_dir, "allocation.json"))
    write_csv(selected_rows, os.path.join(args.out_dir, "selected_components.csv"))
    print("Done!")
    print(f"  {os.path.join(args.out_dir, 'allocation.json')}")
    print(f"  {os.path.join(args.out_dir, 'selected_components.csv')}")
    print(f"  estimated_accept_rate_drop={payload['estimated_accept_rate_drop']:.4f}")


if __name__ == "__main__":
    main()
