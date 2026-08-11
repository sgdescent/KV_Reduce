#!/usr/bin/env python3
"""Evaluate cached KV quantization on real multiple-choice task accuracy."""

from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import os
import random
import re
import zipfile
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

from benchmark_spec_kv_quantization import (
    cached_prefill,
    cached_step,
    crop_cache_to_length,
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
PASSKEY_GENERATOR_V1 = "synthetic_passkey_v1"
PASSKEY_GENERATOR_V2 = "synthetic_passkey_16way_v2"
PASSKEY_GENERATOR_V3 = "synthetic_associative_passkey_v3"
PASSKEY_VARIANT_RANDOM = "random"
PASSKEY_VARIANT_CONFUSABLE = "confusable_records"
LONGBENCH_REPO = "THUDM/LongBench"
LONGBENCH_REVISION = "5e628be450b7e67fb7ae6e201bd6d8f7056f7672"
LONGBENCH_ARCHIVE = "data.zip"
LONGBENCH_RETRIEVAL_FILE = "data/passage_retrieval_en.jsonl"
LONGBENCH_RETRIEVAL_PROMPT = """Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.

{context}

The following is an abstract.

{input}

Please enter the number of the paragraph that the abstract is from. The answer format must be like \"Paragraph 1\", \"Paragraph 2\", etc.

The answer is: """
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
    "longbench_passage_retrieval": {
        "dataset_name": LONGBENCH_REPO,
        "dataset_config": "passage_retrieval_en",
        "split": "test",
        "fewshot_split": None,
        "primary_metric": "normalized_accuracy",
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
    elif task == "longbench_passage_retrieval":
        prompt = LONGBENCH_RETRIEVAL_PROMPT.format(
            context=str(example["context"]),
            input=str(example["input"]),
        )
        paragraph_numbers = [
            int(value)
            for value in re.findall(r"(?m)^Paragraph\s+(\d+):", str(example["context"]))
        ]
        choices = [f" Paragraph {number}" for number in paragraph_numbers]
        answers = [str(answer).strip() for answer in example["answers"]]
        if len(choices) != 30:
            raise ValueError(f"Expected 30 LongBench paragraphs, found {len(choices)}.")
        if not answers or answers[0] not in [choice.strip() for choice in choices]:
            raise ValueError(f"Malformed LongBench retrieval answer: {answers!r}")
        gold = [choice.strip() for choice in choices].index(answers[0])
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
    num_choices: int = 4,
    variant: str = PASSKEY_VARIANT_RANDOM,
) -> List[Tuple[int, Dict[str, Any]]]:
    """Create deterministic passkey retrieval examples."""

    if num_choices < 2:
        raise ValueError("Passkey evaluation requires at least two choices.")
    if variant not in {PASSKEY_VARIANT_RANDOM, PASSKEY_VARIANT_CONFUSABLE}:
        raise ValueError(f"Unsupported passkey variant: {variant!r}")
    depths = (0.1, 0.5, 0.9)
    examples: List[Tuple[int, Dict[str, Any]]] = []
    for source_idx in range(skip_examples, skip_examples + num_examples):
        rng = random.Random(dataset_seed + source_idx * 104_729)
        passkey = f"{rng.randrange(100_000, 1_000_000):06d}"
        choices = {passkey}
        if variant == PASSKEY_VARIANT_CONFUSABLE:
            # Near-collision alternatives prevent the model from succeeding by
            # remembering only a coarse numeric pattern from the target record.
            candidates = []
            for position, original_digit in enumerate(passkey):
                for replacement in "0123456789":
                    if replacement == original_digit:
                        continue
                    candidate = passkey[:position] + replacement + passkey[position + 1 :]
                    if candidate[0] != "0":
                        candidates.append(candidate)
            rng.shuffle(candidates)
            choices.update(candidates[: num_choices - 1])
            if len(choices) < num_choices:
                raise ValueError(
                    f"Confusable passkey supports at most {len(candidates) + 1} choices."
                )
        else:
            while len(choices) < num_choices:
                choices.add(f"{rng.randrange(100_000, 1_000_000):06d}")
        shuffled = sorted(choices)
        rng.shuffle(shuffled)
        example: Dict[str, Any] = {
            "passkey": passkey,
            "choices": shuffled,
            "gold": shuffled.index(passkey),
            "depth": depths[source_idx % len(depths)],
            "variant": variant,
        }
        if variant == PASSKEY_VARIANT_CONFUSABLE:
            tags = set()
            while len(tags) < num_choices:
                tags.add(f"{rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}{rng.randrange(100, 1000)}")
            shuffled_tags = sorted(tags)
            rng.shuffle(shuffled_tags)
            target_tag = shuffled_tags[0]
            distractor_codes = [choice for choice in shuffled if choice != passkey]
            distractor_tags = shuffled_tags[1:]
            records = [
                {"tag": tag, "code": code}
                for tag, code in zip(distractor_tags, distractor_codes)
            ]
            rng.shuffle(records)
            example.update(
                {
                    "target_tag": target_tag,
                    "distractor_records": records,
                }
            )
        examples.append(
            (
                source_idx,
                example,
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
    passkey_num_choices: int = 4,
    passkey_variant: str = PASSKEY_VARIANT_RANDOM,
) -> List[Tuple[int, Dict[str, Any]]]:
    if task == "passkey":
        return generate_passkey_examples(
            num_examples=num_examples,
            skip_examples=skip_examples,
            dataset_seed=dataset_seed,
            num_choices=passkey_num_choices,
            variant=passkey_variant,
        )

    if task == "longbench_passage_retrieval":
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError("huggingface_hub is required for LongBench evaluation.") from exc
        archive_path = hf_hub_download(
            repo_id=LONGBENCH_REPO,
            filename=LONGBENCH_ARCHIVE,
            repo_type="dataset",
            revision=LONGBENCH_REVISION,
        )
        with zipfile.ZipFile(archive_path) as archive:
            with archive.open(LONGBENCH_RETRIEVAL_FILE) as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
        indices = list(range(len(rows)))
        random.Random(dataset_seed).shuffle(indices)
        selected = indices[skip_examples : skip_examples + num_examples]
        return [(source_idx, rows[source_idx]) for source_idx in selected]

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


def tokenize_longbench_example(
    tokenizer,
    prompt: str,
    choices: Sequence[str],
    *,
    max_prompt_tokens: int,
    max_choice_tokens: int,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Apply LongBench's middle truncation while preserving prompt instructions."""

    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=True,
        return_tensors="pt",
    ).input_ids
    if int(prompt_ids.shape[1]) > int(max_prompt_tokens):
        left = int(max_prompt_tokens) // 2
        right = int(max_prompt_tokens) - left
        prompt_ids = torch.cat([prompt_ids[:, :left], prompt_ids[:, -right:]], dim=1)
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
        raise ValueError("LongBench tokenization produced an empty prompt or answer choice.")
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


def assemble_confusable_passkey_prompt_ids(
    *,
    prefix_ids: torch.Tensor,
    filler_ids: torch.Tensor,
    target_record_ids: torch.Tensor,
    distractor_record_ids: Sequence[torch.Tensor],
    query_ids: torch.Tensor,
    target_tokens: int,
    depth: float,
) -> torch.Tensor:
    """Place one target among evenly distributed, confusable archive records."""

    if not 0.0 <= float(depth) <= 1.0:
        raise ValueError("Passkey depth must be between zero and one.")
    if filler_ids.numel() == 0:
        raise ValueError("Passkey filler tokenization is empty.")
    records = list(distractor_record_ids)
    target_record_index = int(round(float(depth) * len(records)))
    records.insert(target_record_index, target_record_ids)
    fixed_tokens = int(
        prefix_ids.numel()
        + query_ids.numel()
        + sum(record.numel() for record in records)
    )
    filler_tokens = int(target_tokens) - fixed_tokens
    if filler_tokens < 1:
        raise ValueError(
            f"Passkey context {target_tokens} is too short for {fixed_tokens} fixed tokens."
        )
    repeats = (filler_tokens + int(filler_ids.numel()) - 1) // int(filler_ids.numel())
    filler = filler_ids.repeat(repeats)[:filler_tokens]
    num_segments = len(records) + 1
    base, remainder = divmod(filler_tokens, num_segments)
    segment_lengths = [base + int(idx < remainder) for idx in range(num_segments)]
    pieces = [prefix_ids]
    offset = 0
    for record, segment_length in zip(records, segment_lengths):
        pieces.extend([filler[offset : offset + segment_length], record])
        offset += segment_length
    pieces.extend([filler[offset:], query_ids])
    prompt = torch.cat(pieces, dim=0)
    if int(prompt.numel()) != int(target_tokens):
        raise AssertionError("Confusable passkey prompt assembly produced the wrong length.")
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

    variant = str(example.get("variant", PASSKEY_VARIANT_RANDOM))
    prefix_ids = encode(
        (
            "Read the archive carefully. Each record ID has a different access code. "
            "Return only the code associated with the requested record ID.\n\n"
            if variant == PASSKEY_VARIANT_CONFUSABLE
            else "Read the following archive carefully and remember the pass key when it appears.\n\n"
        ),
        special_tokens=True,
    )
    filler_ids = encode(
        "The archive records routine observations about weather, books, cities, and daily work. "
    )
    if variant == PASSKEY_VARIANT_CONFUSABLE:
        target_tag = str(example["target_tag"])
        key_ids = encode(
            f"\nArchive record {target_tag}: the access code is {example['passkey']}.\n"
        )
        distractor_ids = [
            encode(f"\nArchive record {record['tag']}: the access code is {record['code']}.\n")
            for record in example["distractor_records"]
        ]
        query_ids = encode(
            f"\nEnd of archive. Question: What is the access code for record {target_tag}? Answer:"
        )
        prompt_ids = assemble_confusable_passkey_prompt_ids(
            prefix_ids=prefix_ids,
            filler_ids=filler_ids,
            target_record_ids=key_ids,
            distractor_record_ids=distractor_ids,
            query_ids=query_ids,
            target_tokens=max_prompt_tokens,
            depth=float(example["depth"]),
        )
    else:
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
    if hasattr(cache, "layers"):
        return copy.deepcopy(cache)
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
    if all(int(bits) >= 16 for bits in k_bits) and all(int(bits) >= 16 for bits in v_bits):
        return cache
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
    rewindable = hasattr(prepared_cache, "crop")
    cache = prepared_cache if rewindable else clone_cache(prepared_cache)
    logits = shared_token_logits(prefill_logits, vocab_size)
    cache_len = int(prompt_len)
    score = 0.0
    token_count = int(choice_ids.shape[1])

    try:
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
    finally:
        if rewindable:
            crop_cache_to_length(cache, prompt_len)
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
    parser.add_argument("--passkey_num_choices", type=int, default=4)
    parser.add_argument(
        "--passkey_variant",
        default=PASSKEY_VARIANT_RANDOM,
        choices=[PASSKEY_VARIANT_RANDOM, PASSKEY_VARIANT_CONFUSABLE],
    )
    parser.add_argument("--passkey_score", default="raw", choices=["raw", "normalized"])
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
        passkey_num_choices=args.passkey_num_choices,
        passkey_variant=args.passkey_variant,
    )
    if not examples:
        raise ValueError("No task examples were loaded.")
    fewshot_examples = (
        []
        if args.task in {"passkey", "longbench_passage_retrieval"}
        else load_examples(
            task=args.task,
            split=str(TASK_SPECS[args.task]["fewshot_split"]),
            num_examples=args.num_fewshot,
            skip_examples=0,
            dataset_seed=args.fewshot_seed,
            streaming=args.streaming,
            passkey_num_choices=args.passkey_num_choices,
            passkey_variant=args.passkey_variant,
        )
    )
    fewshot_prefix = build_fewshot_prefix(
        args.task,
        [example for _source_idx, example in fewshot_examples],
    )
    primary_metric = str(TASK_SPECS[args.task]["primary_metric"])
    if args.task == "passkey":
        primary_metric = f"{args.passkey_score}_accuracy"

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
        elif args.task == "longbench_passage_retrieval":
            prompt_ids, choice_ids = tokenize_longbench_example(
                tokenizer,
                prompt,
                choices,
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
                        "passkey_variant": str(example.get("variant", PASSKEY_VARIANT_RANDOM)),
                        "target_tag": str(example.get("target_tag", "")),
                    }
                )
            elif args.task == "longbench_passage_retrieval":
                row.update(
                    {
                        "answer_paragraph": gold + 1,
                        "answer_depth": (gold + 0.5) / len(choices),
                        "longbench_id": str(example.get("_id", "")),
                        "longbench_length": int(example.get("length", 0)),
                    }
                )
            rows.append(row)
            primary_key = primary_metric.replace("accuracy", "correct")
            running_correct[name].append(float(row[primary_key]))
        if run is not None and ((local_idx + 1) % 10 == 0 or local_idx + 1 == len(examples)):
            run.log(
                {
                    f"task/{name}/running_primary_accuracy": mean(values)
                    for name, values in running_correct.items()
                },
                step=local_idx + 1,
            )
        baseline_key = primary_metric.replace("accuracy", "correct")
        bar.set_postfix(bf16=f"{mean(r[baseline_key] for r in rows if r['config'] == 'none'):.3f}")
        # Release the previous 32K prefix before allocating the next prefill.
        del prepared_cache
        del prefill
        if str(args.device).startswith("cuda"):
            torch.cuda.empty_cache()

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
        elif args.task == "longbench_passage_retrieval":
            for label, lower, upper in (
                ("early", 0.0, 1.0 / 3.0),
                ("middle", 1.0 / 3.0, 2.0 / 3.0),
                ("late", 2.0 / 3.0, 1.0),
            ):
                depth_rows = [
                    row
                    for row in config_rows
                    if lower <= float(row["answer_depth"]) < upper
                    or (label == "late" and float(row["answer_depth"]) == upper)
                ]
                summary[f"depth_{label}/normalized_accuracy"] = mean(
                    row["normalized_correct"] for row in depth_rows
                )
        summary["primary_accuracy"] = summary[primary_metric]
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
            "choice_cache_mode": "append_then_crop_to_prefix",
            "example_cache_release": "explicit_del_then_empty_cache",
            "quantization_update_mode": "prefill_once_then_new_tokens_only",
            "quantization_mode": "fake_quantized_values_with_estimated_packed_bytes",
            "objective": "multiple_choice_task_accuracy",
            "key_quant_axis": args.key_quant_axis,
            "key_group_size": args.key_group_size,
            "key_residual_length": args.key_residual_length,
            "value_quant_scheme": args.value_quant_scheme,
            "task_generator_version": (
                (
                    PASSKEY_GENERATOR_V3
                    if args.passkey_variant == PASSKEY_VARIANT_CONFUSABLE
                    else (
                        PASSKEY_GENERATOR_V1
                        if args.passkey_num_choices == 4
                        else PASSKEY_GENERATOR_V2
                    )
                )
                if args.task == "passkey"
                else None
            ),
            "dataset_repo": LONGBENCH_REPO if args.task == "longbench_passage_retrieval" else None,
            "dataset_revision": (
                LONGBENCH_REVISION if args.task == "longbench_passage_retrieval" else None
            ),
            "dataset_file": (
                LONGBENCH_RETRIEVAL_FILE
                if args.task == "longbench_passage_retrieval"
                else None
            ),
        },
        "task": args.task,
        "split": split,
        "primary_metric": primary_metric,
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
