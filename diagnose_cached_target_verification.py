#!/usr/bin/env python3
"""Audit cached multi-token target verification against sequential decoding.

The speculative decoder verifies several draft tokens in one target call. In BF16,
that batched call can follow a different kernel path than tokenwise greedy decoding.
This script separates harmless numerical drift from causal-mask or rollback bugs by
checking batched logits, suffix invariance, and cache crop/commit behavior.
"""

import argparse
import json
import math
import os
from typing import Any, Dict, List, Sequence

import torch

from benchmark_spec_kv_quantization import (
    cached_prefill,
    cached_step,
    crop_cache_to_length,
    parse_csv_items,
    shared_token_logits,
    top1_logit_margin,
)
from kv_utils import iter_token_blocks, load_causal_lm, load_tokenizer, set_seed, write_json


def logit_comparison(reference: torch.Tensor, candidate: torch.Tensor) -> Dict[str, float]:
    reference_f = reference.float()
    candidate_f = candidate.float().to(reference_f.device)
    reference_logp = torch.log_softmax(reference_f, dim=-1)
    candidate_logp = torch.log_softmax(candidate_f, dim=-1)
    reference_p = reference_logp.exp()
    delta = candidate_f - reference_f
    return {
        "top1_match": float(reference_f.argmax(dim=-1).item() == candidate_f.argmax(dim=-1).item()),
        "reference_top1": int(reference_f.argmax(dim=-1).item()),
        "candidate_top1": int(candidate_f.argmax(dim=-1).item()),
        "reference_margin": top1_logit_margin(reference_f),
        "candidate_margin": top1_logit_margin(candidate_f),
        "mean_abs_logit_delta": float(delta.abs().mean().item()),
        "max_abs_logit_delta": float(delta.abs().max().item()),
        "kl_reference_to_candidate": float(
            torch.sum(reference_p * (reference_logp - candidate_logp), dim=-1).mean().item()
        ),
    }


@torch.no_grad()
def sequential_proposal(
    *,
    model,
    prompt_ids: torch.Tensor,
    steps: int,
    device: str,
    shared_vocab_size: int,
) -> Dict[str, Any]:
    state = cached_prefill(model, prompt_ids, device)
    logits = shared_token_logits(state["logits"], shared_vocab_size)
    cache = state["cache"]
    cache_len = int(state["cache_len"])
    proposal: List[int] = []
    post_token_logits: List[torch.Tensor] = []
    for _ in range(steps):
        token = int(logits.argmax(dim=-1).item())
        proposal.append(token)
        step = cached_step(
            model=model,
            input_ids=torch.tensor([[token]], dtype=prompt_ids.dtype),
            cache=cache,
            cache_len=cache_len,
            device=device,
        )
        logits = shared_token_logits(step["logits"][:, -1, :], shared_vocab_size)
        post_token_logits.append(logits.detach().cpu())
        cache = step["cache"]
        cache_len = int(step["cache_len"])
    return {
        "proposal": proposal,
        "post_token_logits": post_token_logits,
        "next_token": int(logits.argmax(dim=-1).item()),
    }


@torch.no_grad()
def audit_prompt(
    *,
    model,
    prompt_ids: torch.Tensor,
    draft_steps: int,
    device: str,
    shared_vocab_size: int,
) -> Dict[str, Any]:
    sequential = sequential_proposal(
        model=model,
        prompt_ids=prompt_ids,
        steps=draft_steps,
        device=device,
        shared_vocab_size=shared_vocab_size,
    )
    proposal = sequential["proposal"]

    batch_state = cached_prefill(model, prompt_ids, device)
    batch = cached_step(
        model=model,
        input_ids=torch.tensor([proposal], dtype=prompt_ids.dtype),
        cache=batch_state["cache"],
        cache_len=int(batch_state["cache_len"]),
        device=device,
    )
    batch_logits = shared_token_logits(batch["logits"], shared_vocab_size).detach().cpu()

    full_ids = torch.cat(
        [prompt_ids, torch.tensor([proposal], dtype=prompt_ids.dtype)],
        dim=1,
    ).to(device)
    full_logits = shared_token_logits(model(input_ids=full_ids, use_cache=False).logits, shared_vocab_size)
    prompt_len = int(prompt_ids.shape[1])
    full_post_token_logits = full_logits[:, prompt_len : prompt_len + draft_steps, :].detach().cpu()

    position_rows: List[Dict[str, Any]] = []
    for position in range(draft_steps):
        sequential_logits = sequential["post_token_logits"][position]
        row: Dict[str, Any] = {"position": position, "token": proposal[position]}
        row.update(
            {
                f"batch_vs_sequential/{key}": value
                for key, value in logit_comparison(sequential_logits, batch_logits[:, position, :]).items()
            }
        )
        row.update(
            {
                f"full_vs_sequential/{key}": value
                for key, value in logit_comparison(sequential_logits, full_post_token_logits[:, position, :]).items()
            }
        )
        position_rows.append(row)

    suffix_rows: List[Dict[str, Any]] = []
    for change_from in range(1, draft_steps):
        perturbed = proposal[:]
        for idx in range(change_from, draft_steps):
            perturbed[idx] = (perturbed[idx] + 1 + idx) % shared_vocab_size
        perturbed_state = cached_prefill(model, prompt_ids, device)
        perturbed_batch = cached_step(
            model=model,
            input_ids=torch.tensor([perturbed], dtype=prompt_ids.dtype),
            cache=perturbed_state["cache"],
            cache_len=int(perturbed_state["cache_len"]),
            device=device,
        )
        perturbed_logits = shared_token_logits(perturbed_batch["logits"], shared_vocab_size).detach().cpu()
        for unaffected_position in range(change_from):
            suffix_rows.append(
                {
                    "change_from": change_from,
                    "unaffected_position": unaffected_position,
                    **logit_comparison(
                        batch_logits[:, unaffected_position, :],
                        perturbed_logits[:, unaffected_position, :],
                    ),
                }
            )

    rollback_rows: List[Dict[str, Any]] = []
    for accepted in range(draft_steps):
        verification_tokens = proposal[:]
        verification_tokens[accepted] = (verification_tokens[accepted] + 1) % shared_vocab_size
        for idx in range(accepted + 1, draft_steps):
            verification_tokens[idx] = (verification_tokens[idx] + 1 + idx) % shared_vocab_size

        rollback_state = cached_prefill(model, prompt_ids, device)
        rollback_verify = cached_step(
            model=model,
            input_ids=torch.tensor([verification_tokens], dtype=prompt_ids.dtype),
            cache=rollback_state["cache"],
            cache_len=int(rollback_state["cache_len"]),
            device=device,
        )
        committed_len = int(rollback_state["cache_len"]) + accepted
        cropped = crop_cache_to_length(rollback_verify["cache"], committed_len)
        correction = proposal[accepted]
        committed = cached_step(
            model=model,
            input_ids=torch.tensor([[correction]], dtype=prompt_ids.dtype),
            cache=cropped,
            cache_len=committed_len,
            device=device,
        )
        committed_logits = shared_token_logits(committed["logits"][:, -1, :], shared_vocab_size).detach().cpu()

        fresh_prefix = torch.cat(
            [
                prompt_ids,
                torch.tensor([proposal[:accepted] + [correction]], dtype=prompt_ids.dtype),
            ],
            dim=1,
        )
        fresh = cached_prefill(model, fresh_prefix, device)
        fresh_logits = shared_token_logits(fresh["logits"], shared_vocab_size).detach().cpu()
        rollback_rows.append({"accepted": accepted, **logit_comparison(fresh_logits, committed_logits)})

    return {
        "prompt_len": int(prompt_ids.shape[1]),
        "proposal": proposal,
        "sequential_next_token": sequential["next_token"],
        "positions": position_rows,
        "suffix_invariance": suffix_rows,
        "rollback": rollback_rows,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit target batched verification and cache rollback.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--attn_implementation", choices=["eager", "sdpa", "flash_attention_2"], default="sdpa")
    parser.add_argument("--dataset_name", type=str, default="wikitext")
    parser.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--eval_split", type=str, default="validation")
    parser.add_argument("--eval_split_fallbacks", type=str, default="test,train")
    parser.add_argument("--prompt_len", type=int, default=1024)
    parser.add_argument("--num_prompts", type=int, default=2)
    parser.add_argument("--draft_steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="outputs/target_verification_diagnostic/summary.json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    tokenizer = load_tokenizer(args.model)
    model = load_causal_lm(
        args.model,
        device=args.device,
        dtype_name=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    prompts = [
        block.unsqueeze(0)
        for block in iter_token_blocks(
            tokenizer=tokenizer,
            seq_len=args.prompt_len,
            max_blocks=args.num_prompts,
            dataset_name=args.dataset_name,
            dataset_config=args.dataset_config,
            split=args.eval_split,
            split_fallbacks=parse_csv_items(args.eval_split_fallbacks),
            shuffle=False,
            seed=args.seed,
            streaming=False,
        )
    ]
    audits = [
        audit_prompt(
            model=model,
            prompt_ids=prompt,
            draft_steps=args.draft_steps,
            device=args.device,
            shared_vocab_size=int(tokenizer.vocab_size),
        )
        for prompt in prompts
    ]

    all_position_rows = [row for audit in audits for row in audit["positions"]]
    all_suffix_rows = [row for audit in audits for row in audit["suffix_invariance"]]
    all_rollback_rows = [row for audit in audits for row in audit["rollback"]]
    summary = {
        "config": vars(args),
        "num_prompts": len(audits),
        "batch_vs_sequential_top1_mismatches": sum(
            row["batch_vs_sequential/top1_match"] < 0.5 for row in all_position_rows
        ),
        "full_vs_sequential_top1_mismatches": sum(
            row["full_vs_sequential/top1_match"] < 0.5 for row in all_position_rows
        ),
        "causal_suffix_violations": sum(row["max_abs_logit_delta"] > 0.0 for row in all_suffix_rows),
        "rollback_top1_mismatches": sum(row["top1_match"] < 0.5 for row in all_rollback_rows),
        "max_batch_vs_sequential_logit_delta": max(
            (row["batch_vs_sequential/max_abs_logit_delta"] for row in all_position_rows), default=math.nan
        ),
        "max_causal_suffix_logit_delta": max(
            (row["max_abs_logit_delta"] for row in all_suffix_rows), default=math.nan
        ),
        "max_rollback_logit_delta": max(
            (row["max_abs_logit_delta"] for row in all_rollback_rows), default=math.nan
        ),
        "audits": audits,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    write_json(summary, args.out)
    print(json.dumps({key: value for key, value in summary.items() if key != "audits"}, indent=2))


if __name__ == "__main__":
    main()
