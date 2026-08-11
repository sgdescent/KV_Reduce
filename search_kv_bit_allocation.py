#!/usr/bin/env python3
"""
Build a mixed-precision draft KV allocation from sensitivity profile results.

This is intentionally a lightweight, transparent search. It treats one-at-a-time
profile risks as an additive proxy and chooses the lowest-bit candidate for each
layer/component that stays within configurable per-component and total budgets.
The risk may be speculative acceptance drop, ordinary-LM delta NLL, or another
numeric profile column.
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
    max_component_risk: float,
    risk_field: str,
) -> Dict[str, Any]:
    viable = []
    for row in candidates:
        bits = as_int(row, "bits", FULL_PRECISION_BITS)
        if bits not in allowed_bits:
            continue
        risk = max(0.0, as_float(row, risk_field, 0.0))
        if risk <= max_component_risk:
            viable.append(row)
    if not viable:
        return {
            "bits": FULL_PRECISION_BITS,
            "risk": 0.0,
            "saved_bytes": 0.0,
            "source_candidate": "full_precision_fallback",
        }
    viable.sort(key=lambda row: (as_int(row, "bits", FULL_PRECISION_BITS), as_float(row, risk_field, 0.0)))
    chosen = viable[0]
    return {
        "bits": as_int(chosen, "bits", FULL_PRECISION_BITS),
        "risk": max(0.0, as_float(chosen, risk_field, 0.0)),
        "saved_bytes": candidate_saved_bytes(chosen),
        "source_candidate": chosen.get("candidate", ""),
    }


def safer_replacement(
    candidates: List[Dict[str, str]],
    *,
    current_bits: int,
    current_risk: float,
    allowed_bits: List[int],
    risk_field: str,
) -> Dict[str, Any]:
    safer = []
    for row in candidates:
        bits = as_int(row, "bits", FULL_PRECISION_BITS)
        if bits not in allowed_bits:
            continue
        if bits <= current_bits:
            continue
        if max(0.0, as_float(row, risk_field, 0.0)) >= current_risk:
            continue
        safer.append(row)
    if not safer:
        return {
            "bits": FULL_PRECISION_BITS,
            "risk": 0.0,
            "saved_bytes": 0.0,
            "source_candidate": "full_precision_relax",
        }
    safer.sort(key=lambda row: (as_int(row, "bits", FULL_PRECISION_BITS), as_float(row, risk_field, 0.0)))
    chosen = safer[0]
    return {
        "bits": as_int(chosen, "bits", FULL_PRECISION_BITS),
        "risk": max(0.0, as_float(chosen, risk_field, 0.0)),
        "saved_bytes": candidate_saved_bytes(chosen),
        "source_candidate": chosen.get("candidate", ""),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Search a mixed-precision KV allocation from sensitivity results.")
    parser.add_argument("--profile_csv", type=str, required=True)
    parser.add_argument("--num_layers", type=int, default=None)
    parser.add_argument("--allowed_bits", type=str, default="4,8,16")
    parser.add_argument("--risk_field", type=str, default="accept_rate_drop")
    parser.add_argument("--max_component_drop", type=float, default=0.01)
    parser.add_argument("--max_total_drop", type=float, default=0.05)
    parser.add_argument("--max_component_risk", type=float, default=None)
    parser.add_argument("--max_total_risk", type=float, default=None)
    parser.add_argument(
        "--target_profiled_mean_bits",
        type=float,
        default=None,
        help="Use a fixed mean-bit budget over profiled K/V components instead of a risk budget.",
    )
    parser.add_argument(
        "--target_profiled_saved_bytes",
        type=float,
        default=None,
        help=(
            "Require at least this many metadata-aware saved bytes over profiled "
            "K/V components. This is preferred for equal-memory comparisons."
        ),
    )
    parser.add_argument("--name", type=str, default="sensitivity_aware")
    parser.add_argument("--out_dir", type=str, default="outputs/spec_kv_allocation")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rows = read_csv(args.profile_csv)
    max_component_risk = args.max_component_drop if args.max_component_risk is None else args.max_component_risk
    max_total_risk = args.max_total_drop if args.max_total_risk is None else args.max_total_risk
    allowed_bits = sorted(set(parse_csv_ints(args.allowed_bits)))
    if FULL_PRECISION_BITS not in allowed_bits:
        allowed_bits.append(FULL_PRECISION_BITS)
        allowed_bits = sorted(set(allowed_bits))
    if (
        args.target_profiled_mean_bits is not None
        and args.target_profiled_saved_bytes is not None
    ):
        raise ValueError(
            "Choose either target_profiled_mean_bits or target_profiled_saved_bytes, not both."
        )
    if args.target_profiled_saved_bytes is not None and args.target_profiled_saved_bytes < 0:
        raise ValueError("target_profiled_saved_bytes must be nonnegative.")

    profile_rows = [
        row
        for row in rows
        if row.get("component") in {"k", "v"} and as_int(row, "layer", -1) >= 0
    ]
    if not profile_rows:
        raise ValueError("No layer/component rows found in profile CSV.")
    missing_risk = [row.get("candidate", "<unnamed>") for row in profile_rows if row.get(args.risk_field, "") == ""]
    if missing_risk:
        examples = ", ".join(missing_risk[:3])
        raise ValueError(f"Risk field {args.risk_field!r} is missing for profile rows such as: {examples}")

    num_layers = args.num_layers
    if num_layers is None:
        num_layers = max(as_int(row, "layer", 0) for row in profile_rows) + 1

    grouped: Dict[Tuple[int, str], List[Dict[str, str]]] = defaultdict(list)
    for row in profile_rows:
        grouped[(as_int(row, "layer"), str(row["component"]))].append(row)

    k_bits, v_bits = uniform_bit_lists(num_layers, FULL_PRECISION_BITS, FULL_PRECISION_BITS)
    selections: Dict[Tuple[int, str], Dict[str, Any]] = {
        (layer, component): {
            "bits": FULL_PRECISION_BITS,
            "risk": 0.0,
            "saved_bytes": 0.0,
            "source_candidate": "unprofiled_full_precision",
        }
        for layer in range(num_layers)
        for component in ("k", "v")
    }

    if (
        args.target_profiled_mean_bits is not None
        or args.target_profiled_saved_bytes is not None
    ):
        if (
            args.target_profiled_mean_bits is not None
            and not 2.0 <= args.target_profiled_mean_bits <= float(FULL_PRECISION_BITS)
        ):
            raise ValueError("target_profiled_mean_bits must be between 2 and 16.")
        profiled_keys = sorted(grouped)
        options_by_key: Dict[Tuple[int, str], List[Dict[str, Any]]] = {}
        for key in profiled_keys:
            options: Dict[int, Dict[str, Any]] = {
                FULL_PRECISION_BITS: {
                    "bits": FULL_PRECISION_BITS,
                    "risk": 0.0,
                    "saved_bytes": 0.0,
                    "source_candidate": "profiled_full_precision",
                }
            }
            for row in grouped[key]:
                bits = as_int(row, "bits", FULL_PRECISION_BITS)
                if bits not in allowed_bits:
                    continue
                option = {
                    "bits": bits,
                    "risk": max(0.0, as_float(row, args.risk_field, 0.0)),
                    "saved_bytes": candidate_saved_bytes(row),
                    "source_candidate": row.get("candidate", ""),
                }
                if bits not in options or option["risk"] < options[bits]["risk"]:
                    options[bits] = option
            options_by_key[key] = list(options.values())

        # Exact dynamic programming avoids a greedy search accidentally assigning
        # different packed-cache budgets to the two downstream objectives.
        use_byte_budget = args.target_profiled_saved_bytes is not None
        states: Dict[int, Tuple[float, List[Dict[str, Any]]]] = {0: (0.0, [])}
        for key in profiled_keys:
            next_states: Dict[int, Tuple[float, List[Dict[str, Any]]]] = {}
            for total_cost, (total_risk, path) in states.items():
                for option in options_by_key[key]:
                    option_cost = (
                        int(round(float(option["saved_bytes"])))
                        if use_byte_budget
                        else int(option["bits"])
                    )
                    new_cost = total_cost + option_cost
                    new_risk = total_risk + float(option["risk"])
                    previous = next_states.get(new_cost)
                    if previous is None or new_risk < previous[0]:
                        next_states[new_cost] = (new_risk, path + [option])
            states = next_states

        if use_byte_budget:
            target_total = int(round(float(args.target_profiled_saved_bytes)))
            feasible_totals = [total for total in states if total >= target_total]
            # Use the smallest compression that satisfies the cache-byte budget;
            # risk breaks ties among allocations with exactly the same bytes.
            achieved_total = min(feasible_totals) if feasible_totals else None
        else:
            target_total = int(round(float(args.target_profiled_mean_bits) * len(profiled_keys)))
            feasible_totals = [total for total in states if total <= target_total]
            achieved_total = max(feasible_totals) if feasible_totals else None
        if not feasible_totals:
            budget_name = (
                "target_profiled_saved_bytes"
                if use_byte_budget
                else "target_profiled_mean_bits"
            )
            raise ValueError(f"The available candidates cannot satisfy {budget_name}.")
        _, chosen_path = states[int(achieved_total)]
        for key, option in zip(profiled_keys, chosen_path):
            selections[key] = option
    else:
        for key, candidates in grouped.items():
            selections[key] = choose_initial_candidate(
                candidates,
                allowed_bits=allowed_bits,
                max_component_risk=max_component_risk,
                risk_field=args.risk_field,
            )

    def estimated_risk() -> float:
        return float(sum(selection["risk"] for selection in selections.values()))

    while (
        args.target_profiled_mean_bits is None
        and args.target_profiled_saved_bytes is None
        and estimated_risk() > max_total_risk
    ):
        compressive = [
            (key, selection)
            for key, selection in selections.items()
            if int(selection["bits"]) < FULL_PRECISION_BITS and selection["saved_bytes"] > 0
        ]
        if not compressive:
            break
        compressive.sort(
            key=lambda item: (
                item[1]["risk"] / max(item[1]["saved_bytes"], 1.0),
                item[1]["risk"],
            ),
            reverse=True,
        )
        key, selection = compressive[0]
        replacement = safer_replacement(
            grouped.get(key, []),
            current_bits=int(selection["bits"]),
            current_risk=float(selection["risk"]),
            allowed_bits=allowed_bits,
            risk_field=args.risk_field,
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
                    "risk_field": args.risk_field,
                    "risk": float(selection["risk"]),
                    "saved_bytes": float(selection["saved_bytes"]),
                    "saved_mib": float(selection["saved_bytes"]) / (1024.0**2),
                    "source_candidate": selection["source_candidate"],
                }
            )

    payload = {
        "name": args.name,
        "source_profile_csv": args.profile_csv,
        "allowed_bits": allowed_bits,
        "risk_field": args.risk_field,
        "max_component_risk": max_component_risk,
        "max_total_risk": max_total_risk,
        "target_profiled_mean_bits": args.target_profiled_mean_bits,
        "target_profiled_saved_bytes": args.target_profiled_saved_bytes,
        "achieved_profiled_mean_bits": (
            sum(float(selections[key]["bits"]) for key in grouped) / len(grouped)
            if grouped
            else float(FULL_PRECISION_BITS)
        ),
        "estimated_total_risk": estimated_risk(),
        "estimated_saved_bytes_proxy": float(sum(selection["saved_bytes"] for selection in selections.values())),
        "achieved_profiled_saved_bytes": float(
            sum(selections[key]["saved_bytes"] for key in grouped)
        ),
        "k_bits": k_bits,
        "v_bits": v_bits,
        "layers": [
            {"layer": layer, "k_bits": int(k_bits[layer]), "v_bits": int(v_bits[layer])}
            for layer in range(num_layers)
        ],
    }
    if args.risk_field == "accept_rate_drop":
        payload["estimated_accept_rate_drop"] = payload["estimated_total_risk"]
    write_json(payload, os.path.join(args.out_dir, "allocation.json"))
    write_csv(selected_rows, os.path.join(args.out_dir, "selected_components.csv"))
    print("Done!")
    print(f"  {os.path.join(args.out_dir, 'allocation.json')}")
    print(f"  {os.path.join(args.out_dir, 'selected_components.csv')}")
    print(f"  estimated_total_risk={payload['estimated_total_risk']:.6f} ({args.risk_field})")


if __name__ == "__main__":
    main()
