#!/usr/bin/env python3
"""Build one-layer-at-a-time K/V allocations for passkey sensitivity sweeps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List


FULL_PRECISION_BITS = 16


def parse_layers(spec: str, num_layers: int) -> List[int]:
    value = spec.strip().lower()
    if value == "all":
        return list(range(num_layers))
    if value.startswith("top:"):
        count = int(value.split(":", 1)[1])
        if count <= 0 or count > num_layers:
            raise ValueError(f"Invalid top-layer count {count} for {num_layers} layers.")
        return list(range(num_layers - count, num_layers))
    layers = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not layers:
        raise ValueError("At least one layer must be selected.")
    if layers[0] < 0 or layers[-1] >= num_layers:
        raise ValueError(f"Layer selection {layers} is invalid for {num_layers} layers.")
    return layers


def allocation_payload(
    *,
    name: str,
    num_layers: int,
    selected_layers: List[int],
    base_k_bits: int,
    base_v_bits: int,
    changed_layer: int,
    changed_component: str,
    reduced_bits: int,
) -> dict:
    k_bits = [FULL_PRECISION_BITS] * num_layers
    v_bits = [FULL_PRECISION_BITS] * num_layers
    for layer in selected_layers:
        k_bits[layer] = base_k_bits
        v_bits[layer] = base_v_bits
    if changed_component == "k":
        k_bits[changed_layer] = reduced_bits
    elif changed_component == "v":
        v_bits[changed_layer] = reduced_bits
    else:
        raise ValueError(f"Unsupported component {changed_component!r}.")
    return {
        "name": name,
        "source": "passkey_one_layer_sensitivity",
        "selected_layers": selected_layers,
        "base_k_bits": base_k_bits,
        "base_v_bits": base_v_bits,
        "changed_layer": changed_layer,
        "changed_component": changed_component,
        "reduced_bits": reduced_bits,
        "k_bits": k_bits,
        "v_bits": v_bits,
        "layers": [
            {"layer": layer, "k_bits": k_bits[layer], "v_bits": v_bits[layer]}
            for layer in range(num_layers)
        ],
    }


def prepare_allocations(
    *,
    num_layers: int,
    layer_spec: str,
    base_k_bits: int,
    base_v_bits: int,
    reduced_bits: int,
    out_dir: Path,
) -> dict:
    selected_layers = parse_layers(layer_spec, num_layers)
    if not 1 <= reduced_bits < min(base_k_bits, base_v_bits) <= FULL_PRECISION_BITS:
        raise ValueError("Require 1 <= reduced_bits < base K/V bits <= 16.")
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_name = f"selected_k{base_k_bits}v{base_v_bits}"
    baseline = allocation_payload(
        name=baseline_name,
        num_layers=num_layers,
        selected_layers=selected_layers,
        base_k_bits=base_k_bits,
        base_v_bits=base_v_bits,
        changed_layer=selected_layers[0],
        changed_component="k",
        reduced_bits=base_k_bits,
    )
    baseline["changed_layer"] = None
    baseline["changed_component"] = None
    baseline["reduced_bits"] = None
    baseline_path = out_dir / f"{baseline_name}.json"
    baseline_path.write_text(json.dumps(baseline, indent=2), encoding="utf-8")

    candidates = []
    config_specs = ["none", f"allocation:{baseline_path}"]
    for layer in selected_layers:
        for component in ("k", "v"):
            name = f"layer{layer}_{component}{reduced_bits}"
            payload = allocation_payload(
                name=name,
                num_layers=num_layers,
                selected_layers=selected_layers,
                base_k_bits=base_k_bits,
                base_v_bits=base_v_bits,
                changed_layer=layer,
                changed_component=component,
                reduced_bits=reduced_bits,
            )
            path = out_dir / f"{name}.json"
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            candidates.append(
                {
                    "name": name,
                    "layer": layer,
                    "component": component,
                    "bits": reduced_bits,
                    "path": str(path),
                }
            )
            config_specs.append(f"allocation:{path}")

    manifest = {
        "experiment": "passkey_one_layer_sensitivity_v1",
        "num_layers": num_layers,
        "layer_spec": layer_spec,
        "selected_layers": selected_layers,
        "base_k_bits": base_k_bits,
        "base_v_bits": base_v_bits,
        "reduced_bits": reduced_bits,
        "baseline": baseline_name,
        "baseline_path": str(baseline_path),
        "candidates": candidates,
        "quant_configs": ";".join(config_specs),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out_dir / "quant_configs.txt").write_text(manifest["quant_configs"] + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_layers", type=int, required=True)
    parser.add_argument("--layers", default="top:8")
    parser.add_argument("--base_k_bits", type=int, default=4)
    parser.add_argument("--base_v_bits", type=int, default=4)
    parser.add_argument("--reduced_bits", type=int, default=2)
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare_allocations(
        num_layers=args.num_layers,
        layer_spec=args.layers,
        base_k_bits=args.base_k_bits,
        base_v_bits=args.base_v_bits,
        reduced_bits=args.reduced_bits,
        out_dir=args.out_dir,
    )
    print(manifest["quant_configs"])


if __name__ == "__main__":
    main()
