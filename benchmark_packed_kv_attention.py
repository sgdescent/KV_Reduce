#!/usr/bin/env python3
"""Benchmark real packed KV storage with an intentionally unfused decode path."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F

from packed_kv_cache import pack_kivi_kv, tensor_bytes


EVALUATOR_VERSION = "packed_unfused_attention_v1"


def parse_int_list(value: str) -> List[int]:
    return [int(item) for item in re.split(r"[,;]", value) if item.strip()]


def parse_configs(value: str) -> List[Tuple[str, int, int]]:
    configs = []
    for item in re.split(r"[,;]", value):
        normalized = item.strip().lower()
        if not normalized:
            continue
        if normalized in {"none", "native", "bf16", "full"}:
            configs.append(("none", 16, 16))
            continue
        match = re.fullmatch(r"k(\d+)v(\d+)", normalized)
        if not match:
            raise ValueError(f"Unsupported config: {item!r}")
        configs.append((normalized, int(match.group(1)), int(match.group(2))))
    if not configs:
        raise ValueError("At least one quantization config is required.")
    return configs


def dtype_from_name(name: str) -> torch.dtype:
    normalized = name.lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def gqa_attention(query: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    if query.shape[1] == keys.shape[1]:
        return F.scaled_dot_product_attention(query, keys, values, is_causal=False)
    try:
        return F.scaled_dot_product_attention(
            query,
            keys,
            values,
            is_causal=False,
            enable_gqa=True,
        )
    except TypeError:
        repeat = query.shape[1] // keys.shape[1]
        return F.scaled_dot_product_attention(
            query,
            keys.repeat_interleave(repeat, dim=1),
            values.repeat_interleave(repeat, dim=1),
            is_causal=False,
        )


def cuda_time(
    operation: Callable[[], Any],
    *,
    warmup: int,
    iterations: int,
) -> Dict[str, float]:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    times = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        end.record()
        end.synchronize()
        times.append(float(start.elapsed_time(end)))
    ordered = sorted(times)
    return {
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "p95_ms": ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)],
        "min_ms": min(times),
        "max_ms": max(times),
    }


def peak_delta_bytes(operation: Callable[[], Any], device: torch.device) -> int:
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    baseline = torch.cuda.memory_allocated(device)
    output = operation()
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device)
    del output
    return max(0, int(peak - baseline))


def output_error(reference: torch.Tensor, candidate: torch.Tensor) -> Dict[str, float]:
    reference_f32 = reference.float().reshape(-1)
    candidate_f32 = candidate.float().reshape(-1)
    mse = torch.mean((candidate_f32 - reference_f32).square())
    reference_rms = torch.sqrt(torch.mean(reference_f32.square())).clamp_min(1e-12)
    return {
        "output_cosine": float(F.cosine_similarity(reference_f32, candidate_f32, dim=0).item()),
        "output_relative_rmse": float((torch.sqrt(mse) / reference_rms).item()),
        "output_max_abs_error": float(torch.max(torch.abs(candidate_f32 - reference_f32)).item()),
    }


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--contexts", default="1024;4096;16384;32768")
    parser.add_argument("--configs", default="none;k8v4;k4v8;k4v4;k3v4;k4v3")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--query_heads", type=int, default=12)
    parser.add_argument("--kv_heads", type=int, default=2)
    parser.add_argument("--head_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=28)
    parser.add_argument("--group_size", type=int, default=32)
    parser.add_argument("--residual_length", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--out_dir", default="outputs/packed_kv_attention")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default="kv-reduce")
    parser.add_argument("--wandb_group", default="packed-kv-systems")
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--wandb_entity", default=None)
    return parser


@torch.inference_mode()
def main() -> None:
    args = build_parser().parse_args()
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("This benchmark requires a CUDA device.")
    if args.query_heads % args.kv_heads != 0:
        raise ValueError("query_heads must be divisible by kv_heads.")
    if args.iterations <= 0 or args.warmup < 0:
        raise ValueError("iterations must be positive and warmup non-negative.")

    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    contexts = parse_int_list(args.contexts)
    configs = parse_configs(args.configs)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    wandb_run = None
    if args.wandb:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            name=args.wandb_run_name,
            config=vars(args),
        )

    rows: List[Dict[str, Any]] = []
    step = 0
    for context in contexts:
        shape = (args.batch_size, args.kv_heads, context, args.head_dim)
        keys = torch.randn(shape, device=device, dtype=dtype)
        values = torch.randn(shape, device=device, dtype=dtype)
        query = torch.randn(
            (args.batch_size, args.query_heads, 1, args.head_dim),
            device=device,
            dtype=dtype,
        )
        native_bytes = tensor_bytes(keys) + tensor_bytes(values)
        native_operation = lambda: gqa_attention(query, keys, values)
        native_output = native_operation()
        native_timing = cuda_time(native_operation, warmup=args.warmup, iterations=args.iterations)
        native_peak_delta = peak_delta_bytes(native_operation, device)

        for name, k_bits, v_bits in configs:
            torch.cuda.synchronize(device)
            pack_start = torch.cuda.Event(enable_timing=True)
            pack_end = torch.cuda.Event(enable_timing=True)
            pack_start.record()
            packed_k, packed_v = pack_kivi_kv(
                keys,
                values,
                k_bits=k_bits,
                v_bits=v_bits,
                group_size=args.group_size,
                residual_length=args.residual_length,
            )
            pack_end.record()
            pack_end.synchronize()
            pack_ms = float(pack_start.elapsed_time(pack_end))

            if name == "none":
                operation = native_operation
                unpack_operation = lambda: (keys, values)
            else:
                unpack_operation = lambda: (packed_k.unpack(), packed_v.unpack())

                def operation() -> torch.Tensor:
                    unpacked_k, unpacked_v = unpack_operation()
                    return gqa_attention(query, unpacked_k, unpacked_v)

            output = operation()
            timing = native_timing if name == "none" else cuda_time(
                operation,
                warmup=args.warmup,
                iterations=args.iterations,
            )
            unpack_timing = {"median_ms": 0.0, "p95_ms": 0.0, "mean_ms": 0.0}
            if name != "none":
                unpack_timing = cuda_time(
                    unpack_operation,
                    warmup=args.warmup,
                    iterations=args.iterations,
                )
            persistent_bytes = packed_k.storage_bytes + packed_v.storage_bytes
            row = {
                "context": context,
                "config": name,
                "k_bits": k_bits,
                "v_bits": v_bits,
                "native_cache_bytes_one_layer": native_bytes,
                "packed_cache_bytes_one_layer": persistent_bytes,
                "cache_bytes_saved_one_layer": native_bytes - persistent_bytes,
                "cache_saved_fraction": 1.0 - persistent_bytes / native_bytes,
                "native_cache_bytes_model": native_bytes * args.num_layers,
                "packed_cache_bytes_model": persistent_bytes * args.num_layers,
                "key_payload_bytes": packed_k.payload_bytes,
                "key_metadata_bytes": packed_k.metadata_bytes,
                "value_payload_bytes": packed_v.payload_bytes,
                "value_metadata_bytes": packed_v.metadata_bytes,
                "pack_ms": pack_ms,
                "unpack_median_ms": unpack_timing["median_ms"],
                "unpack_p95_ms": unpack_timing["p95_ms"],
                "decode_mean_ms": timing["mean_ms"],
                "decode_median_ms": timing["median_ms"],
                "decode_p95_ms": timing["p95_ms"],
                "native_decode_median_ms": native_timing["median_ms"],
                "unfused_slowdown_vs_native": timing["median_ms"] / native_timing["median_ms"],
                "transient_peak_delta_bytes": native_peak_delta
                if name == "none"
                else peak_delta_bytes(operation, device),
                **output_error(native_output, output),
            }
            rows.append(row)
            step += 1
            print(json.dumps(row, sort_keys=True))
            if wandb_run is not None:
                wandb_run.log(
                    {
                        f"packed/{key}": value
                        for key, value in row.items()
                        if not isinstance(value, str)
                    },
                    step=step,
                )
            del output, packed_k, packed_v
            torch.cuda.empty_cache()
        del native_output, query, keys, values
        torch.cuda.empty_cache()

    payload = {
        "config": vars(args),
        "runtime": {
            "evaluator_version": EVALUATOR_VERSION,
            "storage_mode": "actual_bit_packed_uint8_payloads",
            "decode_mode": "unfused_unpack_then_torch_sdpa",
            "production_throughput_claim": False,
        },
        "rows": rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(out_dir / "results.csv", rows)
    if wandb_run is not None:
        wandb_run.summary["runtime/evaluator_version"] = EVALUATOR_VERSION
        wandb_run.summary["runtime/production_throughput_claim"] = False
        wandb_run.finish()


if __name__ == "__main__":
    main()
