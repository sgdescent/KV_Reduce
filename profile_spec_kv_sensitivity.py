#!/usr/bin/env python3
"""
Profile per-layer/per-component draft KV sensitivity under speculative decoding.

For each selected draft layer and each component (K or V), this runs cached
speculative decoding with only that component quantized, leaving all other draft
KV tensors full precision. The resulting acceptance drop is used by
search_kv_bit_allocation.py to build a mixed-precision allocation.
"""

import argparse
import csv
import os
from typing import Any, Dict, List, Optional, Sequence

import torch

from acceptance_risk_statistics import paired_drop_statistics, zero_drop_statistics
from benchmark_spec_kv_quantization import (
    estimate_total_kv_memory,
    finish_wandb,
    generate_target_reference_records,
    hard_exit_after_success,
    init_wandb,
    run_one_config,
)
from kv_cache_quantization import (
    AFFINE_QUANT,
    FULL_PRECISION_BITS,
    PER_CHANNEL_AXIS,
    PER_TOKEN_AXIS,
    SYMMETRIC_QUANT,
    parse_csv_ints,
    uniform_bit_lists,
)
from kv_utils import (
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
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_csv_items(value: Optional[str]) -> List[str]:
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_layer_spec(spec: str, num_layers: int) -> List[int]:
    spec = spec.strip().lower()
    if spec == "all":
        return list(range(num_layers))
    if spec.startswith("top:"):
        count = max(0, min(num_layers, int(spec.split(":", 1)[1])))
        return list(range(num_layers - count, num_layers))
    if spec.startswith("bottom:"):
        count = max(0, min(num_layers, int(spec.split(":", 1)[1])))
        return list(range(count))
    if spec.startswith("middle:"):
        count = max(0, min(num_layers, int(spec.split(":", 1)[1])))
        start = max(0, (num_layers - count) // 2)
        return list(range(start, start + count))
    if spec.startswith("list:"):
        spec = spec.split(":", 1)[1]
    layers = [int(item.strip()) for item in spec.split(",") if item.strip()]
    if not layers:
        raise ValueError(f"Unsupported layer spec: {spec}")
    for layer in layers:
        if layer < 0 or layer >= num_layers:
            raise ValueError(f"Layer {layer} is out of range for {num_layers} layers.")
    return sorted(set(layers))


def parse_components(spec: str) -> List[str]:
    # SLURM's --export uses commas as field separators, so launch scripts may
    # encode a component list with semicolons instead.
    components = parse_csv_items(spec.replace(";", ","))
    allowed = {"k", "v", "keys", "values"}
    out = []
    for component in components:
        normalized = component.lower()
        if normalized not in allowed:
            raise ValueError(f"Unsupported component {component!r}. Use k,v.")
        out.append("k" if normalized in {"k", "keys"} else "v")
    if not out:
        raise ValueError("At least one component must be selected.")
    return sorted(set(out))


def wandb_candidate_prompt_offset(candidate_idx: int, num_prompts: int) -> int:
    """Reserve one W&B step after each candidate's prompt-level rows."""
    if candidate_idx < 1:
        raise ValueError("candidate_idx must be at least 1; step zero is the baseline.")
    if num_prompts < 1:
        raise ValueError("num_prompts must be positive.")
    return candidate_idx * (num_prompts + 1)


def wandb_candidate_summary_step(candidate_idx: int, num_prompts: int) -> int:
    return wandb_candidate_prompt_offset(candidate_idx, num_prompts) + num_prompts + 1


def cuda_devices(*devices: str) -> List[int]:
    if not torch.cuda.is_available():
        return []
    out: List[int] = []
    for device in devices:
        if not str(device).startswith("cuda"):
            continue
        index = torch.device(device).index
        out.append(0 if index is None else int(index))
    return sorted(set(out))


def build_candidate_bits(
    *,
    num_layers: int,
    layer: int,
    component: str,
    bits: int,
) -> tuple[List[int], List[int]]:
    k_bits, v_bits = uniform_bit_lists(num_layers, FULL_PRECISION_BITS, FULL_PRECISION_BITS)
    if component == "k":
        k_bits[layer] = int(bits)
    elif component == "v":
        v_bits[layer] = int(bits)
    else:
        raise ValueError(f"Unsupported component: {component}")
    return k_bits, v_bits


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile per-layer K/V quantization sensitivity in SpecDec.")
    parser.add_argument("--big_model", type=str, default="Qwen/Qwen2.5-3B")
    parser.add_argument("--small_model", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--big_device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--small_device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--big_dtype", type=str, default="bf16")
    parser.add_argument("--small_dtype", type=str, default="bf16")
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="sdpa",
        choices=["eager", "sdpa", "flash_attention_2"],
    )
    parser.add_argument("--dataset_name", type=str, default="wikitext")
    parser.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--text_file", type=str, default=None)
    parser.add_argument("--text_column", type=str, default=None)
    parser.add_argument("--eval_split", type=str, default="validation")
    parser.add_argument("--eval_split_fallbacks", type=str, default="test,train")
    parser.add_argument("--stream_eval", action="store_true")
    parser.add_argument("--shuffle_eval", action="store_true")
    parser.add_argument("--prompt_len", type=int, default=1024)
    parser.add_argument("--num_prompts", type=int, default=32)
    parser.add_argument("--warmup_prompts", type=int, default=2)
    parser.add_argument(
        "--skip_prompts",
        type=int,
        default=0,
        help="Skip this many token blocks before selecting warmup and profile prompts.",
    )
    parser.add_argument("--draft_steps", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--layers", type=str, default="top:8")
    parser.add_argument("--components", type=str, default="k,v")
    parser.add_argument("--bits", type=str, default="8,4")
    parser.add_argument("--scale_bits", type=int, default=16)
    parser.add_argument(
        "--key_quant_axis",
        type=str,
        default=PER_TOKEN_AXIS,
        choices=[PER_TOKEN_AXIS, PER_CHANNEL_AXIS],
    )
    parser.add_argument("--key_group_size", type=int, default=32)
    parser.add_argument("--key_residual_length", type=int, default=128)
    parser.add_argument(
        "--value_quant_scheme",
        type=str,
        default=SYMMETRIC_QUANT,
        choices=[SYMMETRIC_QUANT, AFFINE_QUANT],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow_incompatible_tokenizers", action="store_true")
    parser.add_argument("--out_dir", type=str, default="outputs/spec_kv_sensitivity")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="kv-reduce")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    wandb_run = init_wandb(args)

    print("Loading models and tokenizers...")
    big_tokenizer = load_tokenizer(args.big_model)
    small_tokenizer = load_tokenizer(args.small_model)
    compatibility = tokenizer_compatibility_report(big_tokenizer, small_tokenizer)
    tokenizers_compatible = bool(
        compatibility["all_probe_encodings_match"] and compatibility["same_vocab_size"]
    )
    if (not tokenizers_compatible) and (not args.allow_incompatible_tokenizers):
        raise ValueError("Tokenizers appear incompatible. Use --allow_incompatible_tokenizers to override.")
    shared_vocab_size = min(int(big_tokenizer.vocab_size), int(small_tokenizer.vocab_size))

    big_model = load_causal_lm(
        args.big_model,
        device=args.big_device,
        dtype_name=args.big_dtype,
        attn_implementation=args.attn_implementation,
    )
    small_model = load_causal_lm(
        args.small_model,
        device=args.small_device,
        dtype_name=args.small_dtype,
        attn_implementation=args.attn_implementation,
    )

    num_layers = int(small_model.config.num_hidden_layers)
    layers = parse_layer_spec(args.layers, num_layers)
    components = parse_components(args.components)
    bit_values = parse_csv_ints(args.bits)
    print(f"Profiling layers: {layers}")
    print(f"Components: {components}; bits: {bit_values}")

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
        skip_blocks=args.skip_prompts,
    )
    all_prompts = [block.unsqueeze(0) for block in prompt_iter]
    warmup_prompts = all_prompts[: args.warmup_prompts]
    profile_prompts = all_prompts[args.warmup_prompts :]
    if not profile_prompts:
        raise ValueError("No profile prompts were loaded.")

    cuda_device_ids = cuda_devices(args.big_device, args.small_device)
    full_k_bits, full_v_bits = uniform_bit_lists(num_layers, FULL_PRECISION_BITS, FULL_PRECISION_BITS)
    print("Generating cached target references once per prompt...")
    warmup_reference_records = generate_target_reference_records(
        prompts=warmup_prompts,
        big_model=big_model,
        max_new_tokens=args.max_new_tokens,
        big_device=args.big_device,
        shared_vocab_size=shared_vocab_size,
    )
    profile_reference_records = generate_target_reference_records(
        prompts=profile_prompts,
        big_model=big_model,
        max_new_tokens=args.max_new_tokens,
        big_device=args.big_device,
        shared_vocab_size=shared_vocab_size,
    )
    warmup_target_references = [record["tokens"] for record in warmup_reference_records]
    warmup_target_margins = [record["top1_margins"] for record in warmup_reference_records]
    profile_target_references = [record["tokens"] for record in profile_reference_records]
    profile_target_margins = [record["top1_margins"] for record in profile_reference_records]

    if warmup_prompts:
        print("Running baseline warmup...")
        run_one_config(
            config_name="baseline_none",
            prompts=warmup_prompts,
            big_model=big_model,
            small_model=small_model,
            draft_steps=args.draft_steps,
            max_new_tokens=args.max_new_tokens,
            big_device=args.big_device,
            small_device=args.small_device,
            topk=args.topk,
            k_bits=full_k_bits,
            v_bits=full_v_bits,
            target_k_bits=None,
            target_v_bits=None,
            cuda_device_ids=cuda_device_ids,
            wandb_run=None,
            wandb_prefix="warmup",
            wandb_step_offset=0,
            shared_vocab_size=shared_vocab_size,
            key_quant_axis=args.key_quant_axis,
            key_group_size=args.key_group_size,
            key_residual_length=args.key_residual_length,
            value_quant_scheme=args.value_quant_scheme,
            target_token_references=warmup_target_references,
            target_margin_references=warmup_target_margins,
        )

    print("Running full-precision baseline...")
    baseline_result = run_one_config(
        config_name="baseline_none",
        prompts=profile_prompts,
        big_model=big_model,
        small_model=small_model,
        draft_steps=args.draft_steps,
        max_new_tokens=args.max_new_tokens,
        big_device=args.big_device,
        small_device=args.small_device,
        topk=args.topk,
        k_bits=full_k_bits,
        v_bits=full_v_bits,
        target_k_bits=None,
        target_v_bits=None,
        cuda_device_ids=cuda_device_ids,
        wandb_run=wandb_run,
        wandb_prefix="sensitivity",
        wandb_step_offset=0,
        shared_vocab_size=shared_vocab_size,
        key_quant_axis=args.key_quant_axis,
        key_group_size=args.key_group_size,
        key_residual_length=args.key_residual_length,
        value_quant_scheme=args.value_quant_scheme,
        target_token_references=profile_target_references,
        target_margin_references=profile_target_margins,
    )
    baseline_summary = baseline_result["summary"]

    raw_rows: List[Dict[str, Any]] = []
    raw_rows.extend(baseline_result["rows"])
    summary_rows: List[Dict[str, Any]] = [
        {
            "candidate": "baseline_none",
            "layer": -1,
            "component": "none",
            "bits": FULL_PRECISION_BITS,
            "accept_rate": baseline_summary.get("overall_accept_rate", 0.0),
            "accept_rate_drop": 0.0,
            **zero_drop_statistics(),
            "accept_mass": baseline_summary.get("round_accept_mass", 0.0),
            "accept_mass_drop": 0.0,
            **zero_drop_statistics("accept_mass_drop"),
            "round_js": baseline_summary.get("round_js", 0.0),
            "round_top1_match": baseline_summary.get("round_top1_match", 0.0),
            "total_cache_saved_fraction": 0.0,
            "draft_cache_saved_fraction": 0.0,
        }
    ]

    candidate_idx = 1
    for layer in layers:
        for component in components:
            for bits in bit_values:
                name = f"layer{layer}_{component}{bits}"
                k_bits, v_bits = build_candidate_bits(
                    num_layers=num_layers,
                    layer=layer,
                    component=component,
                    bits=bits,
                )
                print(f"Profiling candidate: {name}")
                candidate_step_offset = wandb_candidate_prompt_offset(
                    candidate_idx,
                    len(profile_prompts),
                )
                result = run_one_config(
                    config_name=name,
                    prompts=profile_prompts,
                    big_model=big_model,
                    small_model=small_model,
                    draft_steps=args.draft_steps,
                    max_new_tokens=args.max_new_tokens,
                    big_device=args.big_device,
                    small_device=args.small_device,
                    topk=args.topk,
                    k_bits=k_bits,
                    v_bits=v_bits,
                    target_k_bits=None,
                    target_v_bits=None,
                    cuda_device_ids=cuda_device_ids,
                    wandb_run=wandb_run,
                    wandb_prefix="sensitivity",
                    wandb_step_offset=candidate_step_offset,
                    shared_vocab_size=shared_vocab_size,
                    key_quant_axis=args.key_quant_axis,
                    key_group_size=args.key_group_size,
                    key_residual_length=args.key_residual_length,
                    value_quant_scheme=args.value_quant_scheme,
                    target_token_references=profile_target_references,
                    target_margin_references=profile_target_margins,
                )
                raw_rows.extend(result["rows"])
                memory = estimate_total_kv_memory(
                    big_model=big_model,
                    small_model=small_model,
                    big_dtype=args.big_dtype,
                    small_dtype=args.small_dtype,
                    seq_len=args.prompt_len + args.max_new_tokens,
                    k_bits=k_bits,
                    v_bits=v_bits,
                    scale_bits=args.scale_bits,
                    key_quant_axis=args.key_quant_axis,
                    key_group_size=args.key_group_size,
                    key_residual_length=args.key_residual_length,
                    value_quant_scheme=args.value_quant_scheme,
                )
                candidate_summary = result["summary"]
                paired_risk = paired_drop_statistics(
                    baseline_result["rows"],
                    result["rows"],
                )
                paired_accept_mass_risk = paired_drop_statistics(
                    baseline_result["rows"],
                    result["rows"],
                    metric="round_accept_mass",
                    prefix="accept_mass_drop",
                )
                row = {
                    "candidate": name,
                    "layer": int(layer),
                    "component": component,
                    "bits": int(bits),
                    "accept_rate": candidate_summary.get("overall_accept_rate", 0.0),
                    "accept_rate_drop": baseline_summary.get("overall_accept_rate", 0.0)
                    - candidate_summary.get("overall_accept_rate", 0.0),
                    **paired_risk,
                    "accept_mass": candidate_summary.get("round_accept_mass", 0.0),
                    "accept_mass_drop": baseline_summary.get("round_accept_mass", 0.0)
                    - candidate_summary.get("round_accept_mass", 0.0),
                    **paired_accept_mass_risk,
                    "accepted_per_round": candidate_summary.get("accepted_per_round", 0.0),
                    "accepted_per_round_drop": baseline_summary.get("accepted_per_round", 0.0)
                    - candidate_summary.get("accepted_per_round", 0.0),
                    "round_js": candidate_summary.get("round_js", 0.0),
                    "round_js_delta": candidate_summary.get("round_js", 0.0) - baseline_summary.get("round_js", 0.0),
                    "round_top1_match": candidate_summary.get("round_top1_match", 0.0),
                    "round_top1_delta": candidate_summary.get("round_top1_match", 0.0)
                    - baseline_summary.get("round_top1_match", 0.0),
                    **memory,
                }
                summary_rows.append(row)
                if wandb_run is not None:
                    wandb_run.log(
                        {f"sensitivity_summary/{name}/{key}": value for key, value in row.items() if key != "candidate"},
                        step=wandb_candidate_summary_step(candidate_idx, len(profile_prompts)),
                    )
                candidate_idx += 1

    payload = {
        "config": vars(args),
        "runtime": {
            "evaluator_version": "cached_dynamic_v4",
            "objective": "speculative_acceptance_sensitivity",
            "key_quant_axis": args.key_quant_axis,
            "key_group_size": args.key_group_size,
            "key_residual_length": args.key_residual_length,
            "value_quant_scheme": args.value_quant_scheme,
        },
        "layers": layers,
        "components": components,
        "bits": bit_values,
        "baseline_summary": baseline_summary,
        "summary_rows": summary_rows,
    }
    write_csv(raw_rows, os.path.join(args.out_dir, "raw_prompt_rows.csv"))
    write_csv(summary_rows, os.path.join(args.out_dir, "profile_summary.csv"))
    write_json(payload, os.path.join(args.out_dir, "summary.json"))

    print("Done!")
    print(f"  {os.path.join(args.out_dir, 'raw_prompt_rows.csv')}")
    print(f"  {os.path.join(args.out_dir, 'profile_summary.csv')}")
    print(f"  {os.path.join(args.out_dir, 'summary.json')}")
    finish_wandb(wandb_run)


if __name__ == "__main__":
    main()
    hard_exit_after_success()
