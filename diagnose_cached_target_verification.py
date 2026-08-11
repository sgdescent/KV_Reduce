#!/usr/bin/env python3
"""Audit cached multi-token target verification against sequential decoding.

The speculative decoder verifies several draft tokens in one target call. In BF16,
that batched call can follow a different kernel path than tokenwise greedy decoding.
This script separates harmless numerical drift from causal-mask or rollback bugs by
checking batched logits, suffix invariance, and cache crop/commit behavior.
"""

import argparse
import copy
import itertools
import json
import math
import os
from typing import Any, Dict, List, Sequence

import torch

from benchmark_spec_kv_quantization import (
    cached_prefill,
    cached_step,
    crop_cache_to_length,
    draft_next_logits_from_cache,
    greedy_target_generate_with_margins,
    parse_csv_items,
    shared_token_logits,
    top1_logit_margin,
)
from kv_cache_quantization import PER_TOKEN_AXIS, SYMMETRIC_QUANT
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


@torch.no_grad()
def audit_speculative_prompt(
    *,
    big_model,
    small_model,
    prompt_ids: torch.Tensor,
    draft_steps: int,
    max_new_tokens: int,
    big_device: str,
    small_device: str,
    shared_vocab_size: int,
    reset_target_from_sequential_shadow: bool = False,
) -> Dict[str, Any]:
    """Compare the batched verifier to a tokenwise target on the same live prefix."""
    independent_reference = greedy_target_generate_with_margins(
        big_model=big_model,
        prompt_ids=prompt_ids,
        max_new_tokens=max_new_tokens,
        big_device=big_device,
        shared_vocab_size=shared_vocab_size,
    )
    target_state = cached_prefill(big_model, prompt_ids, big_device)
    target_logits = shared_token_logits(target_state["logits"], shared_vocab_size)
    target_cache = target_state["cache"]
    target_cache_len = int(target_state["cache_len"])

    shadow_state = cached_prefill(big_model, prompt_ids, big_device)
    shadow_logits = shared_token_logits(shadow_state["logits"], shared_vocab_size)
    shadow_cache = shadow_state["cache"]
    shadow_cache_len = int(shadow_state["cache_len"])

    draft_state = cached_prefill(small_model, prompt_ids, small_device)
    draft_logits = shared_token_logits(draft_state["logits"], shared_vocab_size)
    draft_cache = draft_state["cache"]
    draft_cache_len = int(draft_state["cache_len"])
    draft_layers = int(small_model.config.num_hidden_layers)
    full_precision_bits = [16] * draft_layers

    generated: List[int] = []
    decision_rows: List[Dict[str, Any]] = []
    round_idx = 0
    while len(generated) < max_new_tokens:
        round_prefix_len = target_cache_len
        proposal: List[int] = []
        for _ in range(min(draft_steps, max_new_tokens - len(generated))):
            token = int(draft_logits.argmax(dim=-1).item())
            proposal.append(token)
            draft_step = draft_next_logits_from_cache(
                small_model=small_model,
                token=token,
                cache=draft_cache,
                cache_len=draft_cache_len,
                small_device=small_device,
                dtype=prompt_ids.dtype,
                k_bits=full_precision_bits,
                v_bits=full_precision_bits,
                key_quant_axis=PER_TOKEN_AXIS,
                key_group_size=32,
                key_residual_length=0,
                value_quant_scheme=SYMMETRIC_QUANT,
            )
            draft_logits = shared_token_logits(draft_step["logits"], shared_vocab_size)
            draft_cache = draft_step["cache"]
            draft_cache_len = int(draft_step["cache_len"])

        verify = cached_step(
            model=big_model,
            input_ids=torch.tensor([proposal], dtype=prompt_ids.dtype),
            cache=target_cache,
            cache_len=target_cache_len,
            device=big_device,
        )
        verify_logits = shared_token_logits(verify["logits"], shared_vocab_size)
        target_cache = verify["cache"]
        target_cache_len = int(verify["cache_len"])

        accepted = 0
        rejection_comparison: Dict[str, Any] | None = None
        for proposal_idx, token in enumerate(proposal):
            verifier_logits = target_logits if proposal_idx == 0 else verify_logits[:, proposal_idx - 1, :]
            comparison = logit_comparison(shadow_logits.detach().cpu(), verifier_logits.detach().cpu())
            row = {
                "round": round_idx,
                "generated_index": len(generated),
                "decision": "proposal",
                "proposal_index": proposal_idx,
                "proposal_token": token,
                **comparison,
            }
            decision_rows.append(row)
            verifier_token = int(verifier_logits.argmax(dim=-1).item())
            if verifier_token != token:
                rejection_comparison = row
                break

            generated.append(token)
            accepted += 1
            shadow_step = cached_step(
                model=big_model,
                input_ids=torch.tensor([[token]], dtype=prompt_ids.dtype),
                cache=shadow_cache,
                cache_len=shadow_cache_len,
                device=big_device,
            )
            shadow_logits = shared_token_logits(shadow_step["logits"][:, -1, :], shared_vocab_size)
            shadow_cache = shadow_step["cache"]
            shadow_cache_len = int(shadow_step["cache_len"])
            if len(generated) >= max_new_tokens:
                break

        if len(generated) >= max_new_tokens:
            break

        correction_logits = target_logits if accepted == 0 else verify_logits[:, accepted - 1, :]
        correction = int(correction_logits.argmax(dim=-1).item())
        if accepted == len(proposal):
            decision_rows.append(
                {
                    "round": round_idx,
                    "generated_index": len(generated),
                    "decision": "verified_bonus",
                    "proposal_index": accepted,
                    "proposal_token": -1,
                    **logit_comparison(shadow_logits.detach().cpu(), correction_logits.detach().cpu()),
                }
            )
        elif rejection_comparison is not None:
            rejection_comparison["decision"] = "target_correction"

        generated.append(correction)
        shadow_step = cached_step(
            model=big_model,
            input_ids=torch.tensor([[correction]], dtype=prompt_ids.dtype),
            cache=shadow_cache,
            cache_len=shadow_cache_len,
            device=big_device,
        )
        shadow_logits = shared_token_logits(shadow_step["logits"][:, -1, :], shared_vocab_size)
        shadow_cache = shadow_step["cache"]
        shadow_cache_len = int(shadow_step["cache_len"])

        committed_len = round_prefix_len + accepted
        target_cache = crop_cache_to_length(target_cache, committed_len)
        draft_cache = crop_cache_to_length(draft_cache, committed_len)
        target_cache_len = committed_len
        draft_cache_len = committed_len

        if reset_target_from_sequential_shadow:
            target_logits = shadow_logits.clone()
            target_cache = copy.deepcopy(shadow_cache)
            target_cache_len = shadow_cache_len
        else:
            target_commit = cached_step(
                model=big_model,
                input_ids=torch.tensor([[correction]], dtype=prompt_ids.dtype),
                cache=target_cache,
                cache_len=target_cache_len,
                device=big_device,
            )
            target_logits = shared_token_logits(target_commit["logits"][:, -1, :], shared_vocab_size)
            target_cache = target_commit["cache"]
            target_cache_len = int(target_commit["cache_len"])

        draft_commit = draft_next_logits_from_cache(
            small_model=small_model,
            token=correction,
            cache=draft_cache,
            cache_len=draft_cache_len,
            small_device=small_device,
            dtype=prompt_ids.dtype,
            k_bits=full_precision_bits,
            v_bits=full_precision_bits,
            key_quant_axis=PER_TOKEN_AXIS,
            key_group_size=32,
            key_residual_length=0,
            value_quant_scheme=SYMMETRIC_QUANT,
        )
        draft_logits = shared_token_logits(draft_commit["logits"], shared_vocab_size)
        draft_cache = draft_commit["cache"]
        draft_cache_len = int(draft_commit["cache_len"])
        round_idx += 1

    generated = generated[:max_new_tokens]
    reference_tokens = independent_reference["tokens"]
    first_reference_mismatch = next(
        (
            idx
            for idx, (generated_token, reference_token) in enumerate(zip(generated, reference_tokens))
            if generated_token != reference_token
        ),
        -1,
    )
    first_top1_mismatch = next(
        (idx for idx, row in enumerate(decision_rows) if row["top1_match"] < 0.5),
        -1,
    )
    return {
        "generated": generated,
        "independent_reference_tokens": reference_tokens,
        "independent_reference_top1_margins": independent_reference["top1_margins"],
        "matches_independent_target_greedy": generated == reference_tokens,
        "first_independent_reference_mismatch": first_reference_mismatch,
        "first_independent_reference_mismatch_margin": (
            independent_reference["top1_margins"][first_reference_mismatch]
            if first_reference_mismatch >= 0
            else math.nan
        ),
        "num_decisions": len(decision_rows),
        "top1_mismatches": sum(row["top1_match"] < 0.5 for row in decision_rows),
        "first_top1_mismatch": first_top1_mismatch,
        "max_abs_logit_delta": max((row["max_abs_logit_delta"] for row in decision_rows), default=math.nan),
        "decisions": decision_rows,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit target batched verification and cache rollback.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B")
    parser.add_argument("--small_model", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--small_device",
        type=str,
        default=None,
        help="Optional draft-model device; useful for memory-heavy target-cache diagnostics.",
    )
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--attn_implementation", choices=["eager", "sdpa", "flash_attention_2"], default="sdpa")
    parser.add_argument("--dataset_name", type=str, default="wikitext")
    parser.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--eval_split", type=str, default="validation")
    parser.add_argument("--eval_split_fallbacks", type=str, default="test,train")
    parser.add_argument("--stream_eval", action="store_true")
    parser.add_argument("--shuffle_eval", action="store_true")
    parser.add_argument("--prompt_len", type=int, default=1024)
    parser.add_argument("--num_prompts", type=int, default=2)
    parser.add_argument("--skip_prompts", type=int, default=0)
    parser.add_argument("--draft_steps", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--skip_unit_audit", action="store_true")
    parser.add_argument("--reset_target_from_sequential_shadow", action="store_true")
    parser.add_argument("--run_speculative_audit", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="outputs/target_verification_diagnostic/summary.json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    small_device = args.small_device or args.device
    set_seed(args.seed)
    tokenizer = load_tokenizer(args.model)
    model = load_causal_lm(
        args.model,
        device=args.device,
        dtype_name=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    block_iterator = iter_token_blocks(
        tokenizer=tokenizer,
        seq_len=args.prompt_len,
        max_blocks=args.skip_prompts + args.num_prompts,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.eval_split,
        split_fallbacks=parse_csv_items(args.eval_split_fallbacks),
        shuffle=args.shuffle_eval,
        seed=args.seed,
        streaming=args.stream_eval,
    )
    prompts = [
        block.unsqueeze(0)
        for block in itertools.islice(block_iterator, args.skip_prompts, None)
    ]
    audits = [] if args.skip_unit_audit else [
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
    speculative_audits: List[Dict[str, Any]] = []
    if args.run_speculative_audit:
        small_model = load_causal_lm(
            args.small_model,
            device=small_device,
            dtype_name=args.dtype,
            attn_implementation=args.attn_implementation,
        )
        speculative_audits = [
            audit_speculative_prompt(
                big_model=model,
                small_model=small_model,
                prompt_ids=prompt,
                draft_steps=args.draft_steps,
                max_new_tokens=args.max_new_tokens,
                big_device=args.device,
                small_device=small_device,
                shared_vocab_size=min(int(tokenizer.vocab_size), int(small_model.config.vocab_size)),
                reset_target_from_sequential_shadow=args.reset_target_from_sequential_shadow,
            )
            for prompt in prompts
        ]

    summary = {
        "config": vars(args),
        "num_prompts": len(prompts),
        "num_unit_audit_prompts": len(audits),
        "num_speculative_audit_prompts": len(speculative_audits),
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
        "speculative_top1_mismatches": sum(
            audit["top1_mismatches"] for audit in speculative_audits
        ),
        "speculative_independent_greedy_mismatches": sum(
            not audit["matches_independent_target_greedy"] for audit in speculative_audits
        ),
        "speculative_prompts_with_top1_mismatch": sum(
            audit["top1_mismatches"] > 0 for audit in speculative_audits
        ),
        "speculative_prompts_with_independent_greedy_mismatch": sum(
            not audit["matches_independent_target_greedy"] for audit in speculative_audits
        ),
        "audits": audits,
        "speculative_audits": speculative_audits,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    write_json(summary, args.out)
    print(json.dumps({key: value for key, value in summary.items() if key != "audits"}, indent=2))


if __name__ == "__main__":
    main()
