#!/usr/bin/env python3
"""Benchmark direct Triton attention over packed KIVI-style KV payloads."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch

from benchmark_packed_kv_attention import (
    cuda_time,
    dtype_from_name,
    gqa_attention,
    output_error,
    parse_configs,
    parse_int_list,
    peak_delta_bytes,
)
from packed_kv_cache import pack_kivi_kv, tensor_bytes
from triton_packed_kv_attention import packed_kv_decode_attention, theoretical_attention_flops


EVALUATOR_VERSION = "packed_fused_attention_v1"


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--contexts", default="1024;4096;16384;32768")
    parser.add_argument("--configs", default="k8v4;k4v8;k4v4")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--query_heads", type=int, default=12)
    parser.add_argument("--kv_heads", type=int, default=2)
    parser.add_argument("--head_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=28)
    parser.add_argument("--group_size", type=int, default=32)
    parser.add_argument("--residual_length", type=int, default=128)
    parser.add_argument("--block_tokens", type=int, default=32, choices=[16, 32, 64])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--out_dir", default="outputs/fused_packed_kv_attention")
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
        raise RuntimeError("This benchmark requires CUDA.")
    if args.query_heads % args.kv_heads != 0:
        raise ValueError("query_heads must be divisible by kv_heads.")
    configs = parse_configs(args.configs)
    for name, k_bits, v_bits in configs:
        if name == "none" or k_bits not in {4, 8} or v_bits not in {4, 8}:
            raise ValueError("Fused benchmark configs currently require K/V bits in {4, 8}.")

    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    contexts = parse_int_list(args.contexts)
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

        for name, k_bits, v_bits in configs:
            packed_keys, packed_values = pack_kivi_kv(
                keys,
                values,
                k_bits=k_bits,
                v_bits=v_bits,
                group_size=args.group_size,
                residual_length=args.residual_length,
            )

            def unpacked_operation() -> torch.Tensor:
                return gqa_attention(query, packed_keys.unpack(), packed_values.unpack())

            def fused_operation() -> torch.Tensor:
                return packed_kv_decode_attention(
                    query,
                    packed_keys,
                    packed_values,
                    block_tokens=args.block_tokens,
                )

            unpacked_output = unpacked_operation()
            fused_output = fused_operation()
            torch.cuda.synchronize(device)
            unpacked_timing = cuda_time(
                unpacked_operation,
                warmup=args.warmup,
                iterations=args.iterations,
            )
            fused_timing = cuda_time(
                fused_operation,
                warmup=args.warmup,
                iterations=args.iterations,
            )
            persistent_bytes = packed_keys.storage_bytes + packed_values.storage_bytes
            row = {
                "context": context,
                "config": name,
                "k_bits": k_bits,
                "v_bits": v_bits,
                "block_tokens": args.block_tokens,
                "native_cache_bytes_one_layer": native_bytes,
                "packed_cache_bytes_one_layer": persistent_bytes,
                "cache_saved_fraction": 1.0 - persistent_bytes / native_bytes,
                "native_cache_bytes_model": native_bytes * args.num_layers,
                "packed_cache_bytes_model": persistent_bytes * args.num_layers,
                "key_payload_bytes": packed_keys.payload_bytes,
                "key_metadata_bytes": packed_keys.metadata_bytes,
                "value_payload_bytes": packed_values.payload_bytes,
                "value_metadata_bytes": packed_values.metadata_bytes,
                "native_decode_median_ms": native_timing["median_ms"],
                "unfused_decode_median_ms": unpacked_timing["median_ms"],
                "fused_decode_mean_ms": fused_timing["mean_ms"],
                "fused_decode_median_ms": fused_timing["median_ms"],
                "fused_decode_p95_ms": fused_timing["p95_ms"],
                "fused_speedup_vs_unfused": unpacked_timing["median_ms"] / fused_timing["median_ms"],
                "fused_speedup_vs_native": native_timing["median_ms"] / fused_timing["median_ms"],
                "native_effective_cache_gbps": native_bytes / (native_timing["median_ms"] * 1e6),
                "fused_effective_cache_gbps": persistent_bytes / (fused_timing["median_ms"] * 1e6),
                "attention_flops": theoretical_attention_flops(query, context),
                "unfused_transient_peak_delta_bytes": peak_delta_bytes(unpacked_operation, device),
                "fused_transient_peak_delta_bytes": peak_delta_bytes(fused_operation, device),
                **{
                    f"native_{key}": value
                    for key, value in output_error(native_output, fused_output).items()
                },
                **{
                    f"kernel_{key}": value
                    for key, value in output_error(unpacked_output, fused_output).items()
                },
            }
            rows.append(row)
            step += 1
            print(json.dumps(row, sort_keys=True))
            if wandb_run is not None:
                wandb_run.log(
                    {f"fused_packed/{key}": value for key, value in row.items() if not isinstance(value, str)},
                    step=step,
                )
            del packed_keys, packed_values, unpacked_output, fused_output
            torch.cuda.empty_cache()
        del keys, values, query, native_output
        torch.cuda.empty_cache()

    payload = {
        "config": vars(args),
        "runtime": {
            "evaluator_version": EVALUATOR_VERSION,
            "storage_mode": "actual_bit_packed_uint8_payloads",
            "decode_mode": "triton_online_softmax_direct_packed_kv",
            "kernel_scope": "single_token_gqa_decode_microbenchmark",
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

