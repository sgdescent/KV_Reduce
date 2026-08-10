#!/usr/bin/env python3
"""Profile KV quantization under the ordinary language-model quality objective.

The same model and cache representation used as the speculative draft can be
evaluated here with teacher-forced continuations. This removes model-size and
architecture confounds when comparing ordinary quality sensitivity against
speculative acceptance sensitivity.
"""

import argparse
import csv
import os
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

from benchmark_spec_kv_quantization import (
    cached_prefill,
    cached_step,
    finish_wandb,
    hard_exit_after_success,
    init_wandb,
    parse_csv_items,
    quantize_cache_for_next_step,
    shared_token_logits,
)
from kv_cache_quantization import (
    AFFINE_QUANT,
    FULL_PRECISION_BITS,
    PER_CHANNEL_AXIS,
    PER_TOKEN_AXIS,
    SYMMETRIC_QUANT,
    bit_allocation_stats,
    estimate_model_kv_cache_bytes,
    parse_csv_ints,
    parse_quant_config_specs,
    uniform_bit_lists,
)
from kv_utils import distribution_metrics, iter_token_blocks, load_causal_lm, load_tokenizer, set_seed, write_json
from profile_spec_kv_sensitivity import parse_components, parse_layer_spec


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


def mean_dict(rows: Iterable[Dict[str, float]]) -> Dict[str, float]:
    sums: Dict[str, float] = defaultdict(float)
    counts: Dict[str, int] = defaultdict(int)
    for row in rows:
        for key, value in row.items():
            sums[key] += float(value)
            counts[key] += 1
    return {key: sums[key] / counts[key] for key in sums if counts[key]}


def validate_labels(label: torch.Tensor, num_classes: int) -> None:
    min_label = int(label.min().item())
    max_label = int(label.max().item())
    if min_label < 0 or max_label >= num_classes:
        raise ValueError(
            f"Label IDs [{min_label}, {max_label}] fall outside the model output vocabulary [0, {num_classes})."
        )


def build_candidate_bits(
    *,
    num_layers: int,
    layer: int,
    component: str,
    bits: int,
) -> Tuple[List[int], List[int]]:
    k_bits, v_bits = uniform_bit_lists(num_layers, FULL_PRECISION_BITS, FULL_PRECISION_BITS)
    if component == "k":
        k_bits[layer] = int(bits)
    elif component == "v":
        v_bits[layer] = int(bits)
    else:
        raise ValueError(f"Unsupported component: {component}")
    return k_bits, v_bits


def collect_reference_logits(
    *,
    model,
    prompt_ids: torch.Tensor,
    continuation_ids: torch.Tensor,
    device: str,
    vocab_size: int,
) -> Tuple[List[torch.Tensor], Dict[str, float]]:
    state = cached_prefill(model, prompt_ids, device)
    logits = shared_token_logits(state["logits"], vocab_size)
    cache = state["cache"]
    cache_len = int(state["cache_len"])
    reference_logits: List[torch.Tensor] = []
    nll_values: List[float] = []

    for token_idx in range(int(continuation_ids.shape[1])):
        label = continuation_ids[:, token_idx].to(device)
        validate_labels(label, int(logits.shape[-1]))
        reference_logits.append(logits.detach().cpu())
        nll_values.append(float(F.cross_entropy(logits.float(), label).item()))
        if token_idx + 1 >= int(continuation_ids.shape[1]):
            break
        step = cached_step(
            model=model,
            input_ids=continuation_ids[:, token_idx : token_idx + 1],
            cache=cache,
            cache_len=cache_len,
            device=device,
        )
        logits = shared_token_logits(step["logits"][:, -1, :], vocab_size)
        cache = step["cache"]
        cache_len = int(step["cache_len"])

    return reference_logits, {"nll": sum(nll_values) / max(1, len(nll_values))}


def evaluate_quantized_sequence(
    *,
    model,
    prompt_ids: torch.Tensor,
    continuation_ids: torch.Tensor,
    reference_logits: Sequence[torch.Tensor],
    reference_nll: float,
    device: str,
    vocab_size: int,
    topk: int,
    k_bits: Sequence[int],
    v_bits: Sequence[int],
    key_quant_axis: str = PER_TOKEN_AXIS,
    key_group_size: int = 32,
    key_residual_length: int = 128,
    value_quant_scheme: str = SYMMETRIC_QUANT,
) -> Dict[str, float]:
    state = cached_prefill(model, prompt_ids, device)
    logits = shared_token_logits(state["logits"], vocab_size)
    cache = quantize_cache_for_next_step(
        state["cache"],
        k_bits,
        v_bits,
        key_quant_axis=key_quant_axis,
        key_group_size=key_group_size,
        key_residual_length=key_residual_length,
        value_quant_scheme=value_quant_scheme,
    )
    cache_len = int(state["cache_len"])
    token_rows: List[Dict[str, float]] = []

    for token_idx, reference_cpu in enumerate(reference_logits):
        label = continuation_ids[:, token_idx].to(device)
        validate_labels(label, int(logits.shape[-1]))
        reference = reference_cpu.to(device)
        metrics = distribution_metrics(reference, logits, topk=topk)
        quantized_nll = float(F.cross_entropy(logits.float(), label).item())
        token_rows.append({"quantized_nll": quantized_nll, **metrics})
        if token_idx + 1 >= len(reference_logits):
            break
        step = cached_step(
            model=model,
            input_ids=continuation_ids[:, token_idx : token_idx + 1],
            cache=cache,
            cache_len=cache_len,
            device=device,
        )
        logits = shared_token_logits(step["logits"][:, -1, :], vocab_size)
        cache = quantize_cache_for_next_step(
            step["cache"],
            k_bits,
            v_bits,
            new_tokens=1,
            key_quant_axis=key_quant_axis,
            key_group_size=key_group_size,
            key_residual_length=key_residual_length,
            value_quant_scheme=value_quant_scheme,
        )
        cache_len = int(step["cache_len"])

    summary = mean_dict(token_rows)
    summary["native_nll"] = float(reference_nll)
    summary["delta_nll"] = float(summary.get("quantized_nll", reference_nll) - reference_nll)
    summary["perplexity_ratio"] = float(torch.exp(torch.tensor(summary["delta_nll"])).item())
    summary["affected_token_fraction"] = max(0.0, (len(reference_logits) - 1) / max(1, len(reference_logits)))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile per-layer KV sensitivity under ordinary LM quality.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bf16")
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
    parser.add_argument("--continuation_len", type=int, default=32)
    parser.add_argument("--num_sequences", type=int, default=32)
    parser.add_argument(
        "--skip_sequences",
        type=int,
        default=0,
        help="Skip this many token blocks before collecting evaluation sequences.",
    )
    parser.add_argument("--layers", type=str, default="top:8")
    parser.add_argument("--components", type=str, default="k,v")
    parser.add_argument("--bits", type=str, default="8,4")
    parser.add_argument(
        "--quant_configs",
        type=str,
        default=None,
        help="Benchmark configs instead of one-component profiles, e.g. none,allocation:path.json.",
    )
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
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument(
        "--quality_risk_metric",
        type=str,
        default="kl",
        choices=["kl", "js", "delta_nll"],
        help="Nonnegative profile score used by the allocator; NLL is always reported for cross-evaluation.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="outputs/kv_quality_sensitivity")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="kv-reduce")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    return parser


@torch.no_grad()
def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    wandb_run = init_wandb(args)

    tokenizer = load_tokenizer(args.model)
    model = load_causal_lm(
        args.model,
        device=args.device,
        dtype_name=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    num_layers = int(model.config.num_hidden_layers)
    # tokenizer.vocab_size excludes added special tokens for tokenizers such as
    # Qwen, while model.config.vocab_size includes their valid output IDs.
    vocab_size = int(model.config.vocab_size)

    if args.quant_configs:
        configs = parse_quant_config_specs(args.quant_configs, num_layers)
        candidates = [
            {
                "candidate": name,
                "layer": -1,
                "component": "allocation" if name != "none" else "none",
                "bits": -1 if name != "none" else FULL_PRECISION_BITS,
                "k_bits": k_bits,
                "v_bits": v_bits,
                "metadata": metadata,
            }
            for name, k_bits, v_bits, metadata in configs
        ]
        mode = "benchmark"
        layers: List[int] = []
        components: List[str] = []
        bit_values: List[int] = []
    else:
        layers = parse_layer_spec(args.layers, num_layers)
        components = parse_components(args.components)
        bit_values = parse_csv_ints(args.bits)
        full_k, full_v = uniform_bit_lists(num_layers, FULL_PRECISION_BITS, FULL_PRECISION_BITS)
        candidates = [
            {
                "candidate": "baseline_none",
                "layer": -1,
                "component": "none",
                "bits": FULL_PRECISION_BITS,
                "k_bits": full_k,
                "v_bits": full_v,
                "metadata": {},
            }
        ]
        for layer in layers:
            for component in components:
                for bits in bit_values:
                    k_bits, v_bits = build_candidate_bits(
                        num_layers=num_layers,
                        layer=layer,
                        component=component,
                        bits=bits,
                    )
                    candidates.append(
                        {
                            "candidate": f"layer{layer}_{component}{bits}",
                            "layer": layer,
                            "component": component,
                            "bits": bits,
                            "k_bits": k_bits,
                            "v_bits": v_bits,
                            "metadata": {},
                        }
                    )
        mode = "sensitivity"

    seq_iter = iter_token_blocks(
        tokenizer=tokenizer,
        seq_len=args.prompt_len + args.continuation_len,
        max_blocks=args.num_sequences,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.eval_split,
        text_file=args.text_file,
        text_column=args.text_column,
        shuffle=args.shuffle_eval,
        seed=args.seed,
        streaming=args.stream_eval,
        split_fallbacks=parse_csv_items(args.eval_split_fallbacks),
        skip_blocks=args.skip_sequences,
    )
    sequences = [block.unsqueeze(0) for block in seq_iter]
    if not sequences:
        raise ValueError("No evaluation sequences were loaded.")

    print(f"Mode: {mode}; candidates: {len(candidates)}; sequences: {len(sequences)}")
    metric_rows: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    raw_rows: List[Dict[str, Any]] = []
    bar = tqdm(sequences, desc="Quality profile", unit="seq")
    for sequence_idx, sequence in enumerate(bar):
        prompt_ids = sequence[:, : args.prompt_len]
        continuation_ids = sequence[:, args.prompt_len :]
        reference_logits, reference_summary = collect_reference_logits(
            model=model,
            prompt_ids=prompt_ids,
            continuation_ids=continuation_ids,
            device=args.device,
            vocab_size=vocab_size,
        )

        for candidate in candidates:
            name = str(candidate["candidate"])
            if candidate["component"] == "none":
                metrics = {
                    "native_nll": reference_summary["nll"],
                    "quantized_nll": reference_summary["nll"],
                    "delta_nll": 0.0,
                    "quality_risk": 0.0,
                    "perplexity_ratio": 1.0,
                    "kl_p_to_q": 0.0,
                    "kl_q_to_p": 0.0,
                    "js": 0.0,
                    "tv": 0.0,
                    "accept_mass": 1.0,
                    "top1_match": 1.0,
                    f"top{args.topk}_overlap": 1.0,
                    "draft_prob_on_target_top1": 0.0,
                    "target_prob_on_target_top1": 0.0,
                    "affected_token_fraction": max(
                        0.0, (args.continuation_len - 1) / max(1, args.continuation_len)
                    ),
                }
            else:
                metrics = evaluate_quantized_sequence(
                    model=model,
                    prompt_ids=prompt_ids,
                    continuation_ids=continuation_ids,
                    reference_logits=reference_logits,
                    reference_nll=reference_summary["nll"],
                    device=args.device,
                    vocab_size=vocab_size,
                    topk=args.topk,
                    k_bits=candidate["k_bits"],
                    v_bits=candidate["v_bits"],
                    key_quant_axis=args.key_quant_axis,
                    key_group_size=args.key_group_size,
                    key_residual_length=args.key_residual_length,
                    value_quant_scheme=args.value_quant_scheme,
                )
            metric_rows[name].append(metrics)
            raw_rows.append(
                {
                    "sequence_idx": sequence_idx,
                    "candidate": name,
                    "layer": candidate["layer"],
                    "component": candidate["component"],
                    "bits": candidate["bits"],
                    **metrics,
                }
            )

    summary_rows: List[Dict[str, Any]] = []
    summaries: Dict[str, Dict[str, Any]] = {}
    for candidate_idx, candidate in enumerate(candidates):
        name = str(candidate["candidate"])
        metrics = mean_dict(metric_rows[name])
        if args.quality_risk_metric == "kl":
            metrics["quality_risk"] = max(0.0, metrics.get("kl_p_to_q", 0.0))
        elif args.quality_risk_metric == "js":
            metrics["quality_risk"] = max(0.0, metrics.get("js", 0.0))
        else:
            metrics["quality_risk"] = max(0.0, metrics.get("delta_nll", 0.0))
        metrics["perplexity_ratio"] = float(torch.exp(torch.tensor(metrics.get("delta_nll", 0.0))).item())
        memory = estimate_model_kv_cache_bytes(
            config=model.config,
            seq_len=args.prompt_len + args.continuation_len,
            dtype_name=args.dtype,
            k_bits_by_layer=candidate["k_bits"],
            v_bits_by_layer=candidate["v_bits"],
            scale_bits=args.scale_bits,
            key_quant_axis=args.key_quant_axis,
            key_group_size=args.key_group_size,
            key_residual_length=args.key_residual_length,
            value_quant_scheme=args.value_quant_scheme,
        )
        row = {
            "candidate": name,
            "layer": candidate["layer"],
            "component": candidate["component"],
            "bits": candidate["bits"],
            **metrics,
            **memory,
            **{f"allocation/{key}": value for key, value in bit_allocation_stats(candidate["k_bits"], candidate["v_bits"]).items()},
        }
        summary_rows.append(row)
        summaries[name] = row
        if wandb_run is not None:
            wandb_run.log(
                {f"quality_summary/{name}/{key}": value for key, value in row.items() if not isinstance(value, str)},
                step=candidate_idx + 1,
            )

    payload = {
        "config": vars(args),
        "runtime": {
            "evaluator_version": "teacher_forced_cached_v1",
            "cache_reused": True,
            "quantization_update_mode": "prefill_once_then_new_tokens_only",
            "objective": "ordinary_language_model_quality",
            "quality_metric": "teacher_forced_continuation_nll",
            "allocation_risk_metric": args.quality_risk_metric,
            "key_quant_axis": args.key_quant_axis,
            "key_group_size": args.key_group_size,
            "key_residual_length": args.key_residual_length,
            "value_quant_scheme": args.value_quant_scheme,
        },
        "mode": mode,
        "model": args.model,
        "num_layers": num_layers,
        "layers": layers,
        "components": components,
        "bits": bit_values,
        "num_sequences": len(sequences),
        "summaries": summaries,
    }
    write_csv(raw_rows, os.path.join(args.out_dir, "raw_sequence_rows.csv"))
    write_csv(summary_rows, os.path.join(args.out_dir, "profile_summary.csv"))
    write_json(payload, os.path.join(args.out_dir, "summary.json"))

    if wandb_run is not None:
        wandb_run.summary["runtime/evaluator_version"] = "teacher_forced_cached_v1"
        wandb_run.summary["num_sequences"] = len(sequences)
    finish_wandb(wandb_run)

    print("Done!")
    print(f"  {os.path.join(args.out_dir, 'profile_summary.csv')}")
    print(f"  {os.path.join(args.out_dir, 'summary.json')}")


if __name__ == "__main__":
    main()
    hard_exit_after_success()
