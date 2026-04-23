#!/usr/bin/env python3
"""
End-to-end benchmark for native speculative decoding vs absorbed shared-cache decoding.

This intentionally lives separately from eval_absorbed_spec_decode.py:
  - eval_absorbed_spec_decode.py focuses on acceptance / quality.
  - this script measures wall-clock latency, PyTorch peak memory, and estimated KV-cache bytes.

The current absorbed prototype recomputes full prefixes for clarity, so the PyTorch wall-clock
numbers are best treated as an implementation benchmark. The analytical KV-cache estimate is the
cleaner measurement of the intended memory saving from partial/full draft-cache sharing.
"""

import argparse
import atexit
import csv
import os
import time
from typing import Any, Dict, List, Optional, Sequence

import torch

from eval_absorbed_spec_decode import (
    aggregate_rows,
    greedy_speculative_decode,
    greedy_target_generate,
    load_absorbed_state,
    parse_csv_items,
    parse_shared_layer_spec,
)
from kv_utils import (
    get_head_dim,
    get_num_kv_heads,
    iter_token_blocks,
    load_causal_lm,
    load_tokenizer,
    set_seed,
    tokenizer_compatibility_report,
    write_json,
)


def write_csv(rows: List[Dict[str, Any]], path: str) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def init_wandb(args: argparse.Namespace) -> Optional[Any]:
    if not args.wandb:
        return None
    try:
        import wandb
    except ImportError as e:
        raise ImportError("W&B logging requested but wandb is not installed.") from e

    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        entity=args.wandb_entity,
        group=args.wandb_group,
        config=vars(args),
    )

    def _finish_wandb() -> None:
        if run is not None:
            run.finish()

    atexit.register(_finish_wandb)
    return run


def parse_modes(value: str) -> List[str]:
    modes = parse_csv_items(value)
    allowed = {"target", "native", "absorbed"}
    for mode in modes:
        if mode not in allowed:
            raise ValueError(f"Unsupported benchmark mode {mode!r}. Choices: {sorted(allowed)}")
    if not modes:
        raise ValueError("At least one benchmark mode is required.")
    return modes


def dtype_num_bytes(dtype_name: str) -> int:
    normalized = dtype_name.lower()
    if normalized in {"bf16", "bfloat16", "fp16", "float16", "half"}:
        return 2
    if normalized in {"fp32", "float32"}:
        return 4
    raise ValueError(f"Unsupported dtype for memory estimate: {dtype_name}")


def kv_cache_bytes(
    *,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    seq_len: int,
    bytes_per_elem: int,
    components_per_layer: float = 2.0,
) -> float:
    return float(num_layers * components_per_layer * num_kv_heads * seq_len * head_dim * bytes_per_elem)


def estimate_kv_memory(
    *,
    big_model,
    small_model,
    big_dtype: str,
    small_dtype: str,
    seq_len: int,
    shared_layer_indices: Sequence[int],
    shared_variant: str,
) -> Dict[str, float]:
    big_layers = int(big_model.config.num_hidden_layers)
    small_layers = int(small_model.config.num_hidden_layers)
    shared_layers = len(set(shared_layer_indices))
    native_draft_layers = small_layers - shared_layers

    target_bytes = kv_cache_bytes(
        num_layers=big_layers,
        num_kv_heads=get_num_kv_heads(big_model.config),
        head_dim=get_head_dim(big_model.config),
        seq_len=seq_len,
        bytes_per_elem=dtype_num_bytes(big_dtype),
        components_per_layer=2.0,
    )
    draft_native_bytes = kv_cache_bytes(
        num_layers=small_layers,
        num_kv_heads=get_num_kv_heads(small_model.config),
        head_dim=get_head_dim(small_model.config),
        seq_len=seq_len,
        bytes_per_elem=dtype_num_bytes(small_dtype),
        components_per_layer=2.0,
    )
    draft_unshared_bytes = kv_cache_bytes(
        num_layers=native_draft_layers,
        num_kv_heads=get_num_kv_heads(small_model.config),
        head_dim=get_head_dim(small_model.config),
        seq_len=seq_len,
        bytes_per_elem=dtype_num_bytes(small_dtype),
        components_per_layer=2.0,
    )
    if shared_variant == "full":
        draft_absorbed_bytes = draft_unshared_bytes
    elif shared_variant == "k_only":
        # K-only sharing still needs native draft V for shared layers, plus full KV on unshared layers.
        draft_shared_v_only_bytes = kv_cache_bytes(
            num_layers=shared_layers,
            num_kv_heads=get_num_kv_heads(small_model.config),
            head_dim=get_head_dim(small_model.config),
            seq_len=seq_len,
            bytes_per_elem=dtype_num_bytes(small_dtype),
            components_per_layer=1.0,
        )
        draft_absorbed_bytes = draft_unshared_bytes + draft_shared_v_only_bytes
    else:
        raise ValueError(f"Unsupported shared_variant: {shared_variant}")

    native_total = target_bytes + draft_native_bytes
    absorbed_total = target_bytes + draft_absorbed_bytes
    return {
        "seq_len": float(seq_len),
        "target_cache_bytes": target_bytes,
        "native_draft_cache_bytes": draft_native_bytes,
        "absorbed_draft_cache_bytes": draft_absorbed_bytes,
        "native_total_cache_bytes": native_total,
        "absorbed_total_cache_bytes": absorbed_total,
        "cache_bytes_saved": native_total - absorbed_total,
        "cache_fraction_saved": (native_total - absorbed_total) / native_total if native_total > 0 else 0.0,
        "target_cache_mib": target_bytes / (1024.0 ** 2),
        "native_total_cache_mib": native_total / (1024.0 ** 2),
        "absorbed_total_cache_mib": absorbed_total / (1024.0 ** 2),
        "cache_mib_saved": (native_total - absorbed_total) / (1024.0 ** 2),
    }


def cuda_devices(*devices: str) -> List[int]:
    out: List[int] = []
    if not torch.cuda.is_available():
        return out
    for device in devices:
        if not str(device).startswith("cuda"):
            continue
        index = torch.device(device).index
        out.append(0 if index is None else int(index))
    return sorted(set(out))


def sync_cuda(devices: Sequence[int]) -> None:
    for device in devices:
        torch.cuda.synchronize(device)


def reset_cuda_peak(devices: Sequence[int]) -> None:
    for device in devices:
        torch.cuda.reset_peak_memory_stats(device)


def cuda_memory_snapshot(devices: Sequence[int], prefix: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for device in devices:
        label = f"{prefix}/cuda:{device}"
        out[f"{label}/allocated_bytes"] = float(torch.cuda.memory_allocated(device))
        out[f"{label}/reserved_bytes"] = float(torch.cuda.memory_reserved(device))
        out[f"{label}/peak_allocated_bytes"] = float(torch.cuda.max_memory_allocated(device))
        out[f"{label}/peak_reserved_bytes"] = float(torch.cuda.max_memory_reserved(device))
        out[f"{label}/allocated_mib"] = out[f"{label}/allocated_bytes"] / (1024.0 ** 2)
        out[f"{label}/reserved_mib"] = out[f"{label}/reserved_bytes"] / (1024.0 ** 2)
        out[f"{label}/peak_allocated_mib"] = out[f"{label}/peak_allocated_bytes"] / (1024.0 ** 2)
        out[f"{label}/peak_reserved_mib"] = out[f"{label}/peak_reserved_bytes"] / (1024.0 ** 2)
    return out


def flatten_round_metrics(result: Dict[str, Any]) -> Dict[str, float]:
    return {f"round_{key}": float(value) for key, value in result.get("round_metrics", {}).items()}


@torch.no_grad()
def run_one_mode(
    *,
    mode: str,
    prompts: Sequence[torch.Tensor],
    big_model,
    small_model,
    absorbed_state: Optional[Dict[str, Any]],
    draft_steps: int,
    max_new_tokens: int,
    big_device: str,
    small_device: str,
    topk: int,
    shared_layer_indices: Sequence[int],
    shared_variant: str,
    norm_match: str,
    cuda_device_ids: Sequence[int],
    wandb_run: Optional[Any],
    wandb_step_offset: int,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    reset_cuda_peak(cuda_device_ids)
    sync_cuda(cuda_device_ids)
    mode_start = time.perf_counter()

    for prompt_idx, prompt_ids in enumerate(prompts):
        sync_cuda(cuda_device_ids)
        prompt_start = time.perf_counter()
        if mode == "target":
            generated = greedy_target_generate(
                big_model=big_model,
                prompt_ids=prompt_ids,
                max_new_tokens=max_new_tokens,
                big_device=big_device,
            )
            result: Dict[str, Any] = {
                "generated_tokens": generated,
                "proposed_tokens": 0,
                "accepted_tokens": 0,
                "accept_rate": 0.0,
                "accepted_per_round": 0.0,
                "full_accept_round_fraction": 0.0,
                "target_calls": max_new_tokens,
                "draft_calls": 0,
                "num_rounds": max_new_tokens,
                "round_metrics": {},
            }
        elif mode == "native":
            result = greedy_speculative_decode(
                mode_name="native",
                big_model=big_model,
                small_model=small_model,
                prompt_ids=prompt_ids,
                absorbed_state=None,
                draft_steps=draft_steps,
                max_new_tokens=max_new_tokens,
                big_device=big_device,
                small_device=small_device,
                topk=topk,
                shared_layer_indices=[],
                shared_variant="full",
                norm_match="none",
            )
        elif mode == "absorbed":
            if absorbed_state is None:
                raise ValueError("absorbed_state is required for absorbed mode.")
            result = greedy_speculative_decode(
                mode_name="absorbed",
                big_model=big_model,
                small_model=small_model,
                prompt_ids=prompt_ids,
                absorbed_state=absorbed_state,
                draft_steps=draft_steps,
                max_new_tokens=max_new_tokens,
                big_device=big_device,
                small_device=small_device,
                topk=topk,
                shared_layer_indices=shared_layer_indices,
                shared_variant=shared_variant,
                norm_match=norm_match,
            )
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        sync_cuda(cuda_device_ids)
        elapsed_s = time.perf_counter() - prompt_start
        generated_tokens = len(result["generated_tokens"])
        row = {
            "mode": mode,
            "prompt_idx": int(prompt_idx),
            "latency_s": float(elapsed_s),
            "generated_tokens": int(generated_tokens),
            "tokens_per_second": float(generated_tokens / elapsed_s) if elapsed_s > 0 else 0.0,
            "ms_per_generated_token": float(1000.0 * elapsed_s / generated_tokens) if generated_tokens > 0 else 0.0,
            "accept_rate": float(result["accept_rate"]),
            "accepted_per_round": float(result["accepted_per_round"]),
            "full_accept_round_fraction": float(result["full_accept_round_fraction"]),
            "proposed_tokens": int(result["proposed_tokens"]),
            "accepted_tokens": int(result["accepted_tokens"]),
            "target_calls": int(result["target_calls"]),
            "draft_calls": int(result["draft_calls"]),
            "num_rounds": int(result["num_rounds"]),
            **flatten_round_metrics(result),
        }
        rows.append(row)

        if wandb_run is not None:
            wandb_run.log(
                {f"benchmark/{mode}/{key}": value for key, value in row.items() if key not in {"mode", "prompt_idx"}},
                step=wandb_step_offset + prompt_idx + 1,
            )

    sync_cuda(cuda_device_ids)
    total_s = time.perf_counter() - mode_start
    memory = cuda_memory_snapshot(cuda_device_ids, "memory")
    numeric_rows = [{k: v for k, v in row.items() if k not in {"mode"}} for row in rows]
    summary = aggregate_rows(numeric_rows, exclude=["prompt_idx"])
    total_generated = sum(int(row["generated_tokens"]) for row in rows)
    total_proposed = sum(int(row["proposed_tokens"]) for row in rows)
    total_accepted = sum(int(row["accepted_tokens"]) for row in rows)
    summary.update(
        {
            "num_prompts": float(len(rows)),
            "total_latency_s": float(total_s),
            "total_generated_tokens": float(total_generated),
            "overall_tokens_per_second": float(total_generated / total_s) if total_s > 0 else 0.0,
            "overall_ms_per_generated_token": float(1000.0 * total_s / total_generated) if total_generated > 0 else 0.0,
            "overall_accept_rate": float(total_accepted / total_proposed) if total_proposed > 0 else 0.0,
            **memory,
        }
    )

    if wandb_run is not None:
        for key, value in summary.items():
            wandb_run.summary[f"benchmark/{mode}/{key}"] = value

    return {"rows": rows, "summary": summary}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark native speculative decoding vs absorbed shared-cache speculative decoding."
    )
    parser.add_argument("--big_model", type=str, default="Qwen/Qwen2.5-3B")
    parser.add_argument("--small_model", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--translator_path", type=str, required=True)
    parser.add_argument("--big_device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--small_device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--big_dtype", type=str, default="bf16")
    parser.add_argument("--small_dtype", type=str, default="bf16")
    parser.add_argument("--dataset_name", type=str, default="wikitext")
    parser.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--text_file", type=str, default=None)
    parser.add_argument("--text_column", type=str, default=None)
    parser.add_argument("--eval_split", type=str, default="validation")
    parser.add_argument("--eval_split_fallbacks", type=str, default="test,train")
    parser.add_argument("--stream_eval", action="store_true")
    parser.add_argument("--shuffle_eval", action="store_true")
    parser.add_argument("--prompt_len", type=int, default=64)
    parser.add_argument("--num_prompts", type=int, default=200)
    parser.add_argument("--warmup_prompts", type=int, default=5)
    parser.add_argument("--draft_steps", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument(
        "--modes",
        type=str,
        default="native,absorbed",
        help='Comma-separated modes to run. Choices: "target", "native", "absorbed".',
    )
    parser.add_argument(
        "--shared_layers",
        type=str,
        default="all",
        help='Which draft layers use absorbed shared-cache attention. Examples: "all", "top:4", "bottom:4", "none".',
    )
    parser.add_argument(
        "--shared_variant",
        type=str,
        choices=["full", "k_only"],
        default="full",
        help='How shared layers consume values. "full" uses target V plus absorbed O. "k_only" uses target-mapped K but native draft V and native o_proj.',
    )
    parser.add_argument(
        "--norm_match",
        type=str,
        choices=["none", "rms", "std"],
        default="none",
        help="Optional per-token output normalization for absorbed full-sharing layers.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow_incompatible_tokenizers", action="store_true")
    parser.add_argument("--out_dir", type=str, default="outputs/absorbed_spec_benchmark")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="kv-absorbed")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    wandb_run = init_wandb(args)

    modes = parse_modes(args.modes)
    absorbed_state = load_absorbed_state(args.translator_path)
    num_layers = len(absorbed_state["layer_map"])
    shared_layer_indices = parse_shared_layer_spec(args.shared_layers, num_layers)

    print("Loading models and tokenizers...")
    big_tokenizer = load_tokenizer(args.big_model)
    small_tokenizer = load_tokenizer(args.small_model)
    compatibility = tokenizer_compatibility_report(big_tokenizer, small_tokenizer)
    if (not compatibility["all_probe_encodings_match"]) and (not args.allow_incompatible_tokenizers):
        raise ValueError("Tokenizers appear incompatible.")

    big_model = load_causal_lm(args.big_model, device=args.big_device, dtype_name=args.big_dtype, attn_implementation="eager")
    small_model = load_causal_lm(args.small_model, device=args.small_device, dtype_name=args.small_dtype, attn_implementation="eager")
    print(f"Benchmark modes: {modes}")
    print(f"Shared layers ({len(shared_layer_indices)}/{num_layers}): {shared_layer_indices}")
    print(f"Shared variant: {args.shared_variant}")
    print(f"Output norm matching: {args.norm_match}")

    prompt_iter = iter_token_blocks(
        tokenizer=big_tokenizer,
        seq_len=args.prompt_len,
        max_blocks=args.num_prompts + args.warmup_prompts,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.eval_split,
        text_file=args.text_file,
        text_column=args.text_column,
        shuffle=args.shuffle_eval,
        seed=args.seed,
        streaming=args.stream_eval,
        split_fallbacks=parse_csv_items(args.eval_split_fallbacks),
    )
    all_prompts = [block.unsqueeze(0) for block in prompt_iter]
    warmup_prompts = all_prompts[: args.warmup_prompts]
    benchmark_prompts = all_prompts[args.warmup_prompts :]
    if len(benchmark_prompts) == 0:
        raise ValueError("No benchmark prompts were loaded.")

    cuda_device_ids = cuda_devices(args.big_device, args.small_device)
    if warmup_prompts:
        print(f"Running {len(warmup_prompts)} warmup prompts per mode...")
        for mode in modes:
            run_one_mode(
                mode=mode,
                prompts=warmup_prompts,
                big_model=big_model,
                small_model=small_model,
                absorbed_state=absorbed_state,
                draft_steps=args.draft_steps,
                max_new_tokens=args.max_new_tokens,
                big_device=args.big_device,
                small_device=args.small_device,
                topk=args.topk,
                shared_layer_indices=shared_layer_indices,
                shared_variant=args.shared_variant,
                norm_match=args.norm_match,
                cuda_device_ids=cuda_device_ids,
                wandb_run=None,
                wandb_step_offset=0,
            )

    all_rows: List[Dict[str, Any]] = []
    summaries: Dict[str, Dict[str, float]] = {}
    for mode_idx, mode in enumerate(modes):
        print(f"Benchmarking mode: {mode}")
        result = run_one_mode(
            mode=mode,
            prompts=benchmark_prompts,
            big_model=big_model,
            small_model=small_model,
            absorbed_state=absorbed_state,
            draft_steps=args.draft_steps,
            max_new_tokens=args.max_new_tokens,
            big_device=args.big_device,
            small_device=args.small_device,
            topk=args.topk,
            shared_layer_indices=shared_layer_indices,
            shared_variant=args.shared_variant,
            norm_match=args.norm_match,
            cuda_device_ids=cuda_device_ids,
            wandb_run=wandb_run,
            wandb_step_offset=mode_idx * len(benchmark_prompts),
        )
        all_rows.extend(result["rows"])
        summaries[mode] = result["summary"]

    memory_estimate = estimate_kv_memory(
        big_model=big_model,
        small_model=small_model,
        big_dtype=args.big_dtype,
        small_dtype=args.small_dtype,
        seq_len=args.prompt_len + args.max_new_tokens,
        shared_layer_indices=shared_layer_indices,
        shared_variant=args.shared_variant,
    )
    summary = {
        "num_prompts": len(benchmark_prompts),
        "warmup_prompts": len(warmup_prompts),
        "modes": modes,
        "translator_path": args.translator_path,
        "translator_output_routing_source": absorbed_state.get("output_routing_source", "unknown"),
        "shared_layers_spec": args.shared_layers,
        "shared_layer_indices": shared_layer_indices,
        "shared_variant": args.shared_variant,
        "norm_match": args.norm_match,
        "memory_estimate": memory_estimate,
        "mode_summaries": summaries,
    }

    write_csv(all_rows, os.path.join(args.out_dir, "benchmark_rows.csv"))
    write_json(summary, os.path.join(args.out_dir, "summary.json"))

    if wandb_run is not None:
        for key, value in memory_estimate.items():
            wandb_run.summary[f"memory_estimate/{key}"] = value
        wandb_run.summary["num_prompts"] = len(benchmark_prompts)
        wandb_run.summary["warmup_prompts"] = len(warmup_prompts)
        wandb_run.summary["translator_output_routing_source"] = absorbed_state.get("output_routing_source", "unknown")
        wandb_run.summary["shared_layers_spec"] = args.shared_layers
        wandb_run.summary["num_shared_layers"] = len(shared_layer_indices)
        wandb_run.summary["shared_variant"] = args.shared_variant
        wandb_run.summary["norm_match"] = args.norm_match

    print("Done!")
    print(f"  {os.path.join(args.out_dir, 'benchmark_rows.csv')}")
    print(f"  {os.path.join(args.out_dir, 'summary.json')}")


if __name__ == "__main__":
    main()
