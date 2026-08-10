#!/usr/bin/env python3
"""Evaluate cached KV quantization on real multiple-choice task accuracy."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import random
import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

from benchmark_spec_kv_quantization import (
    cached_prefill,
    cached_step,
    finish_wandb,
    hard_exit_after_success,
    init_wandb,
    quantize_cache_for_next_step,
    shared_token_logits,
)
from kv_cache_quantization import (
    AFFINE_QUANT,
    PER_CHANNEL_AXIS,
    PER_TOKEN_AXIS,
    SYMMETRIC_QUANT,
    bit_allocation_stats,
    estimate_model_kv_cache_bytes,
    parse_quant_config_specs,
)
from kv_utils import (
    as_legacy_cache,
    clone_legacy_cache,
    legacy_to_cache,
    load_causal_lm,
    load_tokenizer,
    set_seed,
    write_json,
)


EVALUATOR_VERSION = "kv_multiple_choice_cached_v2"
TASK_SPECS = {
    "hellaswag": {
        "dataset_name": "Rowan/hellaswag",
        "dataset_config": None,
        "split": "validation",
        "fewshot_split": "train",
        "primary_metric": "normalized_accuracy",
    },
    "arc_challenge": {
        "dataset_name": "allenai/ai2_arc",
        "dataset_config": "ARC-Challenge",
        "split": "validation",
        "fewshot_split": "train",
        "primary_metric": "raw_accuracy",
    },
    "passkey": {
        "dataset_name": None,
        "dataset_config": None,
        "split": "synthetic",
        "fewshot_split": "synthetic",
        "primary_metric": "raw_accuracy",
    },
}


def write_csv(rows: Sequence[Dict[str, Any]], path: str) -> None:
    if not rows:
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean(values: Iterable[float]) -> float:
    values = [float(value) for value in values]
    return sum(values) / len(values) if values else float("nan")


def clean_hellaswag_text(text: str) -> str:
    text = str(text).replace(" [title]", ". ")
    text = re.sub(r"\[[^]]*\]", "", text)
    return " ".join(text.strip().split())


def format_task_example(task: str, example: Dict[str, Any]) -> Tuple[str, List[str], int]:
    if task == "hellaswag":
        prompt = clean_hellaswag_text(example["ctx"])
        choices = [" " + clean_hellaswag_text(choice) for choice in example["endings"]]
        gold = int(example["label"])
    elif task == "arc_challenge":
        prompt = f"Question: {str(example['question']).strip()}\nAnswer:"
        choice_data = example["choices"]
        labels = [str(label) for label in choice_data["label"]]
        choices = [" " + str(choice).strip() for choice in choice_data["text"]]
        answer = str(example["answerKey"])
        if answer not in labels:
            raise ValueError(f"ARC answer {answer!r} is absent from labels {labels!r}.")
        gold = labels.index(answer)
    elif task == "passkey":
        prompt = "Retrieve the pass key hidden in the archive."
        choices = [" " + str(choice) for choice in example["choices"]]
        gold = int(example["gold"])
    else:
        raise ValueError(f"Unsupported task: {task!r}")
    if not prompt or len(choices) < 2 or not 0 <= gold < len(choices):
        raise ValueError(f"Malformed {task} example.")
    return prompt, choices, gold


def generate_passkey_examples(
    *,
    num_examples: int,
    skip_examples: int,
    dataset_seed: int,
) -> List[Tuple[int, Dict[str, Any]]]:
    """Create deterministic four-way passkey retrieval examples."""

    depths = (0.1, 0.5, 0.9)
    examples: List[Tuple[int, Dict[str, Any]]] = []
    for source_idx in range(skip_examples, skip_examples + num_examples):
        rng = random.Random(dataset_seed + source_idx * 104_729)
        passkey = f"{rng.randrange(100_000, 1_000_000):06d}"
        choices = {passkey}
        while len(choices) < 4:
            choices.add(f"{rng.randrange(100_000, 1_000_000):06d}")
        shuffled = sorted(choices)
        rng.shuffle(shuffled)
        examples.append(
            (
                source_idx,
                {
                    "passkey": passkey,
                    "choices": shuffled,
                    "gold": shuffled.index(passkey),
                    "depth": depths[source_idx % len(depths)],
                },
            )
        )
    return examples


def load_examples(
    *,
    task: str,
    split: str,
    num_examples: int,
    skip_examples: int,
    dataset_seed: int,
    streaming: bool,
) -> List[Tuple[int, Dict[str, Any]]]:
    if task == "passkey":
        return generate_passkey_examples(
            num_examples=num_examples,
            skip_examples=skip_examples,
            dataset_seed=dataset_seed,
        )

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("The datasets package is required for task evaluation.") from exc

    spec = TASK_SPECS[task]
    dataset = load_dataset(
        spec["dataset_name"],
        spec["dataset_config"],
        split=split,
        streaming=streaming,
    )
    if streaming:
        dataset = dataset.shuffle(buffer_size=10_000, seed=dataset_seed)
        iterator = itertools.islice(iter(dataset), skip_examples, skip_examples + num_examples)
        return [(skip_examples + idx, row) for idx, row in enumerate(iterator)]

    dataset = dataset.shuffle(seed=dataset_seed)
    stop = min(len(dataset), skip_examples + num_examples)
    return [(idx, dataset[idx]) for idx in range(skip_examples, stop)]


def build_fewshot_prefix(task: str, examples: Sequence[Dict[str, Any]]) -> str:
    demonstrations = []
    for example in examples:
        prompt, choices, gold = format_task_example(task, example)
        demonstrations.append(prompt + choices[gold])
    return "\n\n".join(demonstrations) + ("\n\n" if demonstrations else "")


def tokenize_example(
    tokenizer,
    prompt: str,
    choices: Sequence[str],
    *,
    max_prompt_tokens: int,
    max_choice_tokens: int,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    old_side = tokenizer.truncation_side
    tokenizer.truncation_side = "left"
    try:
        prompt_ids = tokenizer(
            prompt,
            add_special_tokens=True,
            truncation=True,
            max_length=max_prompt_tokens,
            return_tensors="pt",
        ).input_ids
    finally:
        tokenizer.truncation_side = old_side
    choice_ids = [
        tokenizer(
            choice,
            add_special_tokens=False,
            truncation=True,
            max_length=max_choice_tokens,
            return_tensors="pt",
        ).input_ids
        for choice in choices
    ]
    if prompt_ids.shape[1] == 0 or any(ids.shape[1] == 0 for ids in choice_ids):
        raise ValueError("Tokenization produced an empty prompt or answer choice.")
    return prompt_ids, choice_ids


def assemble_passkey_prompt_ids(
    *,
    prefix_ids: torch.Tensor,
    filler_ids: torch.Tensor,
    key_ids: torch.Tensor,
    query_ids: torch.Tensor,
    target_tokens: int,
    depth: float,
) -> torch.Tensor:
    """Assemble an exact-length prompt with the key at a controlled depth."""

    fixed_tokens = int(prefix_ids.numel() + key_ids.numel() + query_ids.numel())
    filler_tokens = int(target_tokens) - fixed_tokens
    if filler_tokens < 1:
        raise ValueError(
            f"Passkey context {target_tokens} is too short for {fixed_tokens} fixed tokens."
        )
    if not 0.0 <= float(depth) <= 1.0:
        raise ValueError("Passkey depth must be between zero and one.")
    if filler_ids.numel() == 0:
        raise ValueError("Passkey filler tokenization is empty.")
    repeats = (filler_tokens + int(filler_ids.numel()) - 1) // int(filler_ids.numel())
    filler = filler_ids.repeat(repeats)[:filler_tokens]
    before = int(round(float(depth) * filler_tokens))
    prompt = torch.cat(
        [prefix_ids, filler[:before], key_ids, filler[before:], query_ids], dim=0
    )
    if int(prompt.numel()) != int(target_tokens):
        raise AssertionError("Passkey prompt assembly produced the wrong length.")
    return prompt.unsqueeze(0)


def tokenize_passkey_example(
    tokenizer,
    example: Dict[str, Any],
    *,
    max_prompt_tokens: int,
    max_choice_tokens: int,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    def encode(text: str, *, special_tokens: bool = False) -> torch.Tensor:
        return tokenizer(
            text,
            add_special_tokens=special_tokens,
            return_tensors="pt",
        ).input_ids[0]

    prefix_ids = encode(
        "Read the following archive carefully and remember the pass key when it appears.\n\n",
        special_tokens=True,
    )
    filler_ids = encode(
        "The archive records routine observations about weather, books, cities, and daily work. "
    )
    key_ids = encode(f"\nImportant record: the pass key is {example['passkey']}.\n")
    query_ids = encode("\nEnd of archive. Question: What is the pass key? Answer:")
    prompt_ids = assemble_passkey_prompt_ids(
        prefix_ids=prefix_ids,
        filler_ids=filler_ids,
        key_ids=key_ids,
        query_ids=query_ids,
        target_tokens=max_prompt_tokens,
        depth=float(example["depth"]),
    )
    choice_ids = [
        tokenizer(
            " " + str(choice),
            add_special_tokens=False,
            truncation=True,
            max_length=max_choice_tokens,
            return_tensors="pt",
        ).input_ids
        for choice in example["choices"]
    ]
    if any(ids.shape[1] == 0 for ids in choice_ids):
        raise ValueError("Passkey tokenization produced an empty answer choice.")
    return prompt_ids, choice_ids


def clone_cache(cache):
    return legacy_to_cache(clone_legacy_cache(as_legacy_cache(cache)))


def prepare_prefill_cache(
    cache,
    *,
    k_bits: Sequence[int],
    v_bits: Sequence[int],
    key_quant_axis: str,
    key_group_size: int,
    key_residual_length: int,
    value_quant_scheme: str,
):
    return quantize_cache_for_next_step(
        clone_cache(cache),
        k_bits,
        v_bits,
        key_quant_axis=key_quant_axis,
        key_group_size=key_group_size,
        key_residual_length=key_residual_length,
        value_quant_scheme=value_quant_scheme,
    )


@torch.no_grad()
def score_choice_from_prefill(
    *,
    model,
    prefill_logits: torch.Tensor,
    prepared_cache,
    prompt_len: int,
    choice_ids: torch.Tensor,
    device: str,
    vocab_size: int,
    k_bits: Sequence[int],
    v_bits: Sequence[int],
    key_quant_axis: str,
    key_group_size: int,
    key_residual_length: int,
    value_quant_scheme: str,
) -> Tuple[float, int]:
    cache = clone_cache(prepared_cache)
    logits = shared_token_logits(prefill_logits, vocab_size)
    cache_len = int(prompt_len)
    score = 0.0
    token_count = int(choice_ids.shape[1])

    for token_idx in range(token_count):
        token = choice_ids[:, token_idx].to(device)
        if int(token.max().item()) >= int(logits.shape[-1]):
            raise ValueError("Choice token falls outside the model output vocabulary.")
        score += float(F.log_softmax(logits.float(), dim=-1).gather(-1, token[:, None]).item())
        if token_idx + 1 >= token_count:
            break
        step = cached_step(
            model=model,
            input_ids=choice_ids[:, token_idx : token_idx + 1],
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
    return score, token_count


def choose_answer(scores: Sequence[float]) -> int:
    if not scores:
        raise ValueError("At least one answer score is required.")
    return max(range(len(scores)), key=lambda idx: float(scores[idx]))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--attn_implementation", default="sdpa", choices=["eager", "sdpa", "flash_attention_2"])
    parser.add_argument("--task", default="hellaswag", choices=sorted(TASK_SPECS))
    parser.add_argument("--split", default=None)
    parser.add_argument("--num_examples", type=int, default=256)
    parser.add_argument("--skip_examples", type=int, default=0)
    parser.add_argument("--dataset_seed", type=int, default=1729)
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num_fewshot", type=int, default=8)
    parser.add_argument("--fewshot_seed", type=int, default=31415)
    parser.add_argument("--max_prompt_tokens", type=int, default=1024)
    parser.add_argument("--max_choice_tokens", type=int, default=128)
    parser.add_argument("--quant_configs", default="none;k8v4;k4v8;k4v4;k3v4;k4v3")
    parser.add_argument("--scale_bits", type=int, default=16)
    parser.add_argument("--key_quant_axis", default=PER_CHANNEL_AXIS, choices=[PER_TOKEN_AXIS, PER_CHANNEL_AXIS])
    parser.add_argument("--key_group_size", type=int, default=32)
    parser.add_argument("--key_residual_length", type=int, default=128)
    parser.add_argument("--value_quant_scheme", default=AFFINE_QUANT, choices=[SYMMETRIC_QUANT, AFFINE_QUANT])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", default="outputs/kivi_multiple_choice")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default="kv-reduce")
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--wandb_group", default=None)
    return parser


@torch.no_grad()
def main() -> None:
    args = build_parser().parse_args()
    if args.num_examples <= 0 or args.skip_examples < 0:
        raise ValueError("num_examples must be positive and skip_examples must be nonnegative.")
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    run = init_wandb(args)

    tokenizer = load_tokenizer(args.model)
    model = load_causal_lm(
        args.model,
        device=args.device,
        dtype_name=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    vocab_size = int(model.config.vocab_size)
    configs = parse_quant_config_specs(args.quant_configs, int(model.config.num_hidden_layers))
    split = args.split or str(TASK_SPECS[args.task]["split"])
    examples = load_examples(
        task=args.task,
        split=split,
        num_examples=args.num_examples,
        skip_examples=args.skip_examples,
        dataset_seed=args.dataset_seed,
        streaming=args.streaming,
    )
    if not examples:
        raise ValueError("No task examples were loaded.")
    fewshot_examples = (
        []
        if args.task == "passkey"
        else load_examples(
            task=args.task,
            split=str(TASK_SPECS[args.task]["fewshot_split"]),
            num_examples=args.num_fewshot,
            skip_examples=0,
            dataset_seed=args.fewshot_seed,
            streaming=args.streaming,
        )
    )
    fewshot_prefix = build_fewshot_prefix(
        args.task,
        [example for _source_idx, example in fewshot_examples],
    )

    rows: List[Dict[str, Any]] = []
    prompt_lengths: List[int] = []
    choice_lengths: List[int] = []
    evaluation_lengths: List[int] = []
    running_correct: Dict[str, List[float]] = defaultdict(list)
    bar = tqdm(examples, desc=f"{args.task} examples", unit="example")
    for local_idx, (source_idx, example) in enumerate(bar):
        prompt, choices, gold = format_task_example(args.task, example)
        prompt = fewshot_prefix + prompt
        if args.task == "passkey":
            prompt_ids, choice_ids = tokenize_passkey_example(
                tokenizer,
                example,
                max_prompt_tokens=args.max_prompt_tokens,
                max_choice_tokens=args.max_choice_tokens,
            )
        else:
            prompt_ids, choice_ids = tokenize_example(
                tokenizer,
                prompt,
                choices,
                max_prompt_tokens=args.max_prompt_tokens,
                max_choice_tokens=args.max_choice_tokens,
            )
        prompt_len = int(prompt_ids.shape[1])
        prompt_lengths.append(prompt_len)
        choice_lengths.extend(int(ids.shape[1]) for ids in choice_ids)
        evaluation_lengths.extend(prompt_len + int(ids.shape[1]) for ids in choice_ids)
        prefill = cached_prefill(model, prompt_ids, args.device)

        example_results: Dict[str, Dict[str, Any]] = {}
        for name, k_bits, v_bits, _metadata in configs:
            prepared_cache = prepare_prefill_cache(
                prefill["cache"],
                k_bits=k_bits,
                v_bits=v_bits,
                key_quant_axis=args.key_quant_axis,
                key_group_size=args.key_group_size,
                key_residual_length=args.key_residual_length,
                value_quant_scheme=args.value_quant_scheme,
            )
            raw_scores: List[float] = []
            normalized_scores: List[float] = []
            for ids in choice_ids:
                raw_score, token_count = score_choice_from_prefill(
                    model=model,
                    prefill_logits=prefill["logits"],
                    prepared_cache=prepared_cache,
                    prompt_len=prompt_len,
                    choice_ids=ids,
                    device=args.device,
                    vocab_size=vocab_size,
                    k_bits=k_bits,
                    v_bits=v_bits,
                    key_quant_axis=args.key_quant_axis,
                    key_group_size=args.key_group_size,
                    key_residual_length=args.key_residual_length,
                    value_quant_scheme=args.value_quant_scheme,
                )
                raw_scores.append(raw_score)
                normalized_scores.append(raw_score / max(1, token_count))
            raw_prediction = choose_answer(raw_scores)
            normalized_prediction = choose_answer(normalized_scores)
            example_results[name] = {
                "raw_prediction": raw_prediction,
                "normalized_prediction": normalized_prediction,
                "raw_correct": float(raw_prediction == gold),
                "normalized_correct": float(normalized_prediction == gold),
                "gold_raw_score": raw_scores[gold],
                "gold_normalized_score": normalized_scores[gold],
                "raw_scores": raw_scores,
                "normalized_scores": normalized_scores,
            }

        baseline = example_results.get("none")
        if baseline is None:
            raise ValueError("quant_configs must include the BF16 'none' baseline.")
        for name, _k_bits, _v_bits, _metadata in configs:
            result = example_results[name]
            row = {
                "task": args.task,
                "seed": args.seed,
                "example_idx": local_idx,
                "source_idx": source_idx,
                "config": name,
                "gold": gold,
                "num_choices": len(choices),
                "prompt_tokens": prompt_len,
                "mean_choice_tokens": mean(ids.shape[1] for ids in choice_ids),
                "raw_prediction": result["raw_prediction"],
                "normalized_prediction": result["normalized_prediction"],
                "raw_correct": result["raw_correct"],
                "normalized_correct": result["normalized_correct"],
                "raw_agrees_with_bf16": float(result["raw_prediction"] == baseline["raw_prediction"]),
                "normalized_agrees_with_bf16": float(
                    result["normalized_prediction"] == baseline["normalized_prediction"]
                ),
                "gold_raw_score": result["gold_raw_score"],
                "gold_normalized_score": result["gold_normalized_score"],
                "raw_scores": json.dumps(result["raw_scores"]),
                "normalized_scores": json.dumps(result["normalized_scores"]),
            }
            if args.task == "passkey":
                row.update(
                    {
                        "passkey_depth": float(example["depth"]),
                        "passkey": str(example["passkey"]),
                    }
                )
            rows.append(row)
            primary_key = "normalized_correct" if args.task == "hellaswag" else "raw_correct"
            running_correct[name].append(float(row[primary_key]))
        if run is not None and ((local_idx + 1) % 10 == 0 or local_idx + 1 == len(examples)):
            run.log(
                {
                    f"task/{name}/running_primary_accuracy": mean(values)
                    for name, values in running_correct.items()
                },
                step=local_idx + 1,
            )
        baseline_key = "normalized_correct" if args.task == "hellaswag" else "raw_correct"
        bar.set_postfix(bf16=f"{mean(r[baseline_key] for r in rows if r['config'] == 'none'):.3f}")

    memory_seq_len = int(round(mean(evaluation_lengths)))
    peak_memory_seq_len = max(evaluation_lengths)
    summaries: Dict[str, Dict[str, Any]] = {}
    for name, k_bits, v_bits, metadata in configs:
        config_rows = [row for row in rows if row["config"] == name]
        memory_rows = [
            estimate_model_kv_cache_bytes(
                config=model.config,
                seq_len=seq_len,
                dtype_name=args.dtype,
                k_bits_by_layer=k_bits,
                v_bits_by_layer=v_bits,
                scale_bits=args.scale_bits,
                key_quant_axis=args.key_quant_axis,
                key_group_size=args.key_group_size,
                key_residual_length=args.key_residual_length,
                value_quant_scheme=args.value_quant_scheme,
            )
            for seq_len in evaluation_lengths
        ]
        memory = {
            key: mean(row[key] for row in memory_rows)
            for key in memory_rows[0]
        }
        peak_memory = estimate_model_kv_cache_bytes(
            config=model.config,
            seq_len=peak_memory_seq_len,
            dtype_name=args.dtype,
            k_bits_by_layer=k_bits,
            v_bits_by_layer=v_bits,
            scale_bits=args.scale_bits,
            key_quant_axis=args.key_quant_axis,
            key_group_size=args.key_group_size,
            key_residual_length=args.key_residual_length,
            value_quant_scheme=args.value_quant_scheme,
        )
        summary = {
            "raw_accuracy": mean(row["raw_correct"] for row in config_rows),
            "normalized_accuracy": mean(row["normalized_correct"] for row in config_rows),
            "raw_agreement_with_bf16": mean(row["raw_agrees_with_bf16"] for row in config_rows),
            "normalized_agreement_with_bf16": mean(
                row["normalized_agrees_with_bf16"] for row in config_rows
            ),
            "mean_gold_raw_score": mean(row["gold_raw_score"] for row in config_rows),
            "mean_gold_normalized_score": mean(row["gold_normalized_score"] for row in config_rows),
            **memory,
            **{f"peak_{key}": value for key, value in peak_memory.items()},
            **{f"allocation/{key}": value for key, value in bit_allocation_stats(k_bits, v_bits).items()},
            **{f"metadata/{key}": value for key, value in metadata.items() if isinstance(value, (str, int, float, bool))},
        }
        if args.task == "passkey":
            for depth in sorted({float(row["passkey_depth"]) for row in config_rows}):
                depth_rows = [
                    row for row in config_rows if float(row["passkey_depth"]) == depth
                ]
                summary[f"depth_{int(round(100 * depth))}/accuracy"] = mean(
                    row["raw_correct"] for row in depth_rows
                )
        summary["primary_accuracy"] = summary[str(TASK_SPECS[args.task]["primary_metric"])]
        summaries[name] = summary
        if run is not None:
            for key, value in summary.items():
                if isinstance(value, (int, float)):
                    run.summary[f"task/{name}/{key}"] = value

    payload = {
        "config": vars(args),
        "runtime": {
            "evaluator_version": EVALUATOR_VERSION,
            "cache_reused": True,
            "quantization_update_mode": "prefill_once_then_new_tokens_only",
            "quantization_mode": "fake_quantized_values_with_estimated_packed_bytes",
            "objective": "multiple_choice_task_accuracy",
            "key_quant_axis": args.key_quant_axis,
            "key_group_size": args.key_group_size,
            "key_residual_length": args.key_residual_length,
            "value_quant_scheme": args.value_quant_scheme,
            "task_generator_version": (
                "synthetic_passkey_v1" if args.task == "passkey" else None
            ),
        },
        "task": args.task,
        "split": split,
        "primary_metric": TASK_SPECS[args.task]["primary_metric"],
        "num_examples": len(examples),
        "num_fewshot": len(fewshot_examples),
        "source_index_range": [examples[0][0], examples[-1][0]],
        "mean_prompt_tokens": mean(prompt_lengths),
        "mean_choice_tokens": mean(choice_lengths),
        "memory_seq_len": memory_seq_len,
        "peak_memory_seq_len": peak_memory_seq_len,
        "summaries": summaries,
    }
    write_csv(rows, os.path.join(args.out_dir, "example_rows.csv"))
    write_json(payload, os.path.join(args.out_dir, "summary.json"))
    if run is not None:
        run.summary["runtime/evaluator_version"] = EVALUATOR_VERSION
        run.summary["num_examples"] = len(examples)
    finish_wandb(run)
    print(os.path.join(args.out_dir, "summary.json"))


if __name__ == "__main__":
    main()
    hard_exit_after_success()
