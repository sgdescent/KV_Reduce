#!/usr/bin/env python3
"""Select a retrieval-aware mixed K/V policy from layer-sensitivity results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def write_allocation(path: Path, payload: Dict[str, Any]) -> None:
    payload["layers"] = [
        {"layer": layer, "k_bits": payload["k_bits"][layer], "v_bits": payload["v_bits"][layer]}
        for layer in range(len(payload["k_bits"]))
    ]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def select_policy(
    *,
    sensitivity_summary: Path,
    sensitivity_manifest: Path,
    num_reductions: int,
    out_dir: Path,
) -> Dict[str, Any]:
    summary = json.loads(sensitivity_summary.read_text(encoding="utf-8"))
    manifest = json.loads(sensitivity_manifest.read_text(encoding="utf-8"))
    if not summary.get("complete_gate"):
        raise ValueError("Layer-sensitivity completeness gate did not pass.")
    if summary.get("selected_layers") != manifest.get("selected_layers"):
        raise ValueError("Sensitivity summary and allocation manifest select different layers.")
    baseline = json.loads(Path(manifest["baseline_path"]).read_text(encoding="utf-8"))
    selected_layers = [int(layer) for layer in manifest["selected_layers"]]
    candidates: List[Dict[str, Any]] = []
    for row in summary["layer_results"]:
        layer = int(row["layer"])
        for component in ("k", "v"):
            candidates.append(
                {
                    "layer": layer,
                    "component": component,
                    "harm_mean": float(row[f"{component}_harm_mean"]),
                    "harm_ci_low": float(row[f"{component}_harm_ci_low"]),
                    "harm_ci_high": float(row[f"{component}_harm_ci_high"]),
                }
            )
    if num_reductions <= 0 or num_reductions > len(candidates):
        raise ValueError(f"num_reductions must be in [1, {len(candidates)}].")
    ranked = sorted(
        candidates,
        key=lambda row: (
            row["harm_ci_high"],
            row["harm_mean"],
            row["layer"],
            row["component"],
        ),
    )
    chosen = ranked[:num_reductions]
    reduced_bits = int(manifest["reduced_bits"])
    out_dir.mkdir(parents=True, exist_ok=True)

    controls = {}
    for name, reduce_component in (("all_k2v4", "k"), ("all_k4v2", "v")):
        payload = {
            "name": name,
            "source": "passkey_layer_policy_control",
            "selected_layers": selected_layers,
            "k_bits": list(baseline["k_bits"]),
            "v_bits": list(baseline["v_bits"]),
        }
        for layer in selected_layers:
            payload[f"{reduce_component}_bits"][layer] = reduced_bits
        path = out_dir / f"{name}.json"
        write_allocation(path, payload)
        controls[name] = str(path)

    learned = {
        "name": "retrieval_aware_mixed",
        "source": "passkey_layer_sensitivity_upper_ci_selection",
        "profile_summary": str(sensitivity_summary),
        "profile_manifest": str(sensitivity_manifest),
        "selection_metric": "paired_accuracy_harm_ci_high",
        "num_reductions": num_reductions,
        "selected_layers": selected_layers,
        "chosen_reductions": chosen,
        "ranked_candidates": ranked,
        "k_bits": list(baseline["k_bits"]),
        "v_bits": list(baseline["v_bits"]),
    }
    for candidate in chosen:
        learned[f"{candidate['component']}_bits"][int(candidate["layer"])] = reduced_bits
    learned_path = out_dir / "retrieval_aware_mixed.json"
    write_allocation(learned_path, learned)

    baseline_path = out_dir / "selected_k4v4.json"
    baseline_copy = dict(baseline)
    baseline_copy["name"] = "selected_k4v4"
    baseline_copy["source"] = "passkey_layer_policy_baseline"
    write_allocation(baseline_path, baseline_copy)
    quant_configs = ";".join(
        [
            "none",
            f"allocation:{baseline_path}",
            f"allocation:{controls['all_k2v4']}",
            f"allocation:{controls['all_k4v2']}",
            f"allocation:{learned_path}",
        ]
    )
    selection = {
        "experiment": "passkey_retrieval_aware_policy_v1",
        "complete_profile_gate": True,
        "profile_source_index_ranges": summary.get("source_index_ranges"),
        "num_reductions": num_reductions,
        "chosen_reductions": chosen,
        "controls": controls,
        "baseline": str(baseline_path),
        "learned": str(learned_path),
        "quant_configs": quant_configs,
    }
    (out_dir / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    (out_dir / "quant_configs.txt").write_text(quant_configs + "\n", encoding="utf-8")
    return selection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensitivity_summary", type=Path, required=True)
    parser.add_argument("--sensitivity_manifest", type=Path, required=True)
    parser.add_argument("--num_reductions", type=int, default=8)
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()
    result = select_policy(
        sensitivity_summary=args.sensitivity_summary,
        sensitivity_manifest=args.sensitivity_manifest,
        num_reductions=args.num_reductions,
        out_dir=args.out_dir,
    )
    print(json.dumps(result["chosen_reductions"], indent=2))


if __name__ == "__main__":
    main()
