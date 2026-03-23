#!/usr/bin/env python3
import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import torch

from kv_utils import (
    as_legacy_cache,
    distribution_metrics,
    first_mismatch_position,
    iter_token_blocks,
    legacy_to_cache,
    load_causal_lm,
    load_tokenizer,
    set_seed,
    write_json,
    write_jsonl,
)



def parse_float_list(text: str) -> List[float]:
    return [float(x) for x in text.split(",") if x.strip()]



def parse_str_list(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]



def get_layer_specs(layer_mode: str, layers_text: Optional[str], num_layers: int) -> List[Tuple[str, Optional[List[int]]]]:
    if layer_mode == "all":
        return [("all", None)]
    if layer_mode == "single":
        return [(f"layer_{i}", [i]) for i in range(num_layers)]
    if layer_mode == "custom":
        if not layers_text:
            raise ValueError("--layers is required when --layer_mode=custom")
        layers = [int(x) for x in layers_text.split(",") if x.strip()]
        for idx in layers:
            if idx < 0 or idx >= num_layers:
                raise ValueError(f"Layer index {idx} out of range [0, {num_layers - 1}]")
        return [("custom_" + "-".join(map(str, layers)), layers)]
    raise ValueError(f"Unknown layer_mode: {layer_mode}")



def perturb_legacy_cache(
    legacy_cache,
    alpha: float,
    perturb_target: str,
    layer_indices: Optional[List[int]],
    relative_rms: bool,
):
    perturbed = []
    selected = None if layer_indices is None else set(layer_indices)
    for layer_idx, layer_cache in enumerate(legacy_cache):
        k, v = layer_cache[0], layer_cache[1]
        new_k = k.clone()
        new_v = v.clone()
        if selected is None or layer_idx in selected:
            if perturb_target in ("keys", "both"):
                scale = new_k.float().pow(2).mean().sqrt() if relative_rms else torch.tensor(1.0, device=new_k.device)
                new_k = new_k + alpha * scale.to(dtype=new_k.dtype, device=new_k.device) * torch.randn_like(new_k)
            if perturb_target in ("values", "both"):
                scale = new_v.float().pow(2).mean().sqrt() if relative_rms else torch.tensor(1.0, device=new_v.device)
                new_v = new_v + alpha * scale.to(dtype=new_v.dtype, device=new_v.device) * torch.randn_like(new_v)
        perturbed.append((new_k, new_v))
    return tuple(perturbed)


@torch.no_grad()
def greedy_continue_from_state(model, first_logits: torch.Tensor, past_key_values, max_new_tokens: int) -> List[int]:
    if max_new_tokens <= 0:
        return []
    next_token = first_logits.argmax(dim=-1, keepdim=True)
    if next_token.shape[0] != 1:
        raise ValueError("greedy_continue_from_state assumes batch size 1")
    generated = [int(next_token.item())]
    cache = past_key_values
    current = next_token
    for _ in range(max_new_tokens - 1):
        out = model(input_ids=current, past_key_values=cache, use_cache=True)
        current = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(int(current.item()))
        cache = out.past_key_values
    return generated



def aggregate_rows(rows: List[Dict], metric_names: List[str]) -> List[Dict]:
    grouped = defaultdict(list)
    for row in rows:
        key = (row["alpha"], row["perturb_target"], row["layer_spec"])
        grouped[key].append(row)
    summary = []
    for (alpha, perturb_target, layer_spec), items in grouped.items():
        agg = {
            "alpha": alpha,
            "perturb_target": perturb_target,
            "layer_spec": layer_spec,
            "num_examples": len(items),
        }
        for metric_name in metric_names:
            values = [float(item[metric_name]) for item in items if metric_name in item]
            agg[metric_name] = float(sum(values) / len(values)) if values else float("nan")
        summary.append(agg)
    summary.sort(key=lambda x: (x["layer_spec"], x["perturb_target"], x["alpha"]))
    return summary



def write_csv(rows: List[Dict], path: str) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)



def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep KV-cache perturbations and measure logit/token stability.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--dataset_name", type=str, default="wikitext")
    parser.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--text_file", type=str, default=None)
    parser.add_argument("--text_column", type=str, default=None)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--num_sequences", type=int, default=128)
    parser.add_argument("--alphas", type=str, default="0,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1")
    parser.add_argument("--perturb_targets", type=str, default="keys,values,both")
    parser.add_argument("--layer_mode", type=str, choices=["all", "single", "custom"], default="all")
    parser.add_argument("--layers", type=str, default=None)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--generate_steps", type=int, default=16)
    parser.add_argument("--save_examples", type=int, default=8)
    parser.add_argument("--relative_rms", action="store_true", default=True)
    parser.add_argument("--no_relative_rms", action="store_false", dest="relative_rms")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="outputs/kv_perturbation")
    args = parser.parse_args()

    if args.seq_len < 2:
        raise ValueError("seq_len must be at least 2.")

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    tokenizer = load_tokenizer(args.model)
    model = load_causal_lm(args.model, device=args.device, dtype_name=args.dtype)
    num_layers = int(model.config.num_hidden_layers)

    alphas = parse_float_list(args.alphas)
    perturb_targets = parse_str_list(args.perturb_targets)
    layer_specs = get_layer_specs(args.layer_mode, args.layers, num_layers)

    raw_rows: List[Dict] = []
    example_rows: List[Dict] = []

    block_iter = iter_token_blocks(
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        max_blocks=args.num_sequences,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.split,
        text_file=args.text_file,
        text_column=args.text_column,
        shuffle=args.shuffle,
        seed=args.seed,
    )

    for block_idx, block in enumerate(block_iter):
        ids = block.unsqueeze(0).to(args.device)
        context = ids[:, :-1]
        current = ids[:, -1:]

        # Build decode-position tensors that are identical for every forward call
        # on this block, regardless of which cache object is passed.
        T = context.shape[1]
        cache_position = torch.arange(T, T + current.shape[1], device=args.device)
        attention_mask = torch.ones(
            (ids.shape[0], T + current.shape[1]), dtype=torch.long, device=args.device
        )

        with torch.no_grad():
            context_out = model(input_ids=context, use_cache=True)
            # Convert immediately to the legacy tuple format. Everything that
            # follows — baseline included — will use a cache rebuilt from this
            # representation, so the round-trip is identical for alpha=0 and
            # alpha>0. That makes the alpha=0 row a true identity baseline.
            baseline_cache_legacy = as_legacy_cache(context_out.past_key_values)
            baseline_cache = legacy_to_cache(baseline_cache_legacy)
            baseline_out = model(
                input_ids=current,
                attention_mask=attention_mask,
                past_key_values=baseline_cache,
                use_cache=True,
                cache_position=cache_position,
            )
            baseline_logits = baseline_out.logits[:, -1, :]

        baseline_continuation = greedy_continue_from_state(
            model=model,
            first_logits=baseline_logits,
            past_key_values=baseline_out.past_key_values,
            max_new_tokens=args.generate_steps,
        )

        prompt_text = tokenizer.decode(ids[0].tolist(), skip_special_tokens=False)
        baseline_text = tokenizer.decode(baseline_continuation, skip_special_tokens=False)

        if block_idx % 10 == 0:
            print(f"Processed {block_idx + 1} / {args.num_sequences} sequences")

        for alpha in alphas:
            for perturb_target in perturb_targets:
                for layer_name, layer_indices in layer_specs:
                    pert_legacy = perturb_legacy_cache(
                        legacy_cache=baseline_cache_legacy,
                        alpha=alpha,
                        perturb_target=perturb_target,
                        layer_indices=layer_indices,
                        relative_rms=args.relative_rms,
                    )
                    pert_cache = legacy_to_cache(pert_legacy)
                    with torch.no_grad():
                        pert_out = model(
                            input_ids=current,
                            attention_mask=attention_mask,
                            past_key_values=pert_cache,
                            use_cache=True,
                            cache_position=cache_position,
                        )
                        pert_logits = pert_out.logits[:, -1, :]

                    metrics = distribution_metrics(baseline_logits, pert_logits, topk=args.topk)
                    pert_continuation = greedy_continue_from_state(
                        model=model,
                        first_logits=pert_logits,
                        past_key_values=pert_out.past_key_values,
                        max_new_tokens=args.generate_steps,
                    )
                    mismatch_pos = first_mismatch_position(baseline_continuation, pert_continuation)

                    row = {
                        "example_idx": block_idx,
                        "alpha": alpha,
                        "perturb_target": perturb_target,
                        "layer_spec": layer_name,
                        "base_top1_token": int(baseline_logits.argmax(dim=-1).item()),
                        "pert_top1_token": int(pert_logits.argmax(dim=-1).item()),
                        "gen_exact_match": int(baseline_continuation == pert_continuation),
                        "gen_prefix_match_len": int(mismatch_pos),
                        **metrics,
                    }
                    raw_rows.append(row)

                    if len(example_rows) < args.save_examples:
                        example_rows.append(
                            {
                                "example_idx": block_idx,
                                "alpha": alpha,
                                "perturb_target": perturb_target,
                                "layer_spec": layer_name,
                                "prompt": prompt_text,
                                "baseline_continuation_ids": baseline_continuation,
                                "perturbed_continuation_ids": pert_continuation,
                                "baseline_continuation_text": baseline_text,
                                "perturbed_continuation_text": tokenizer.decode(pert_continuation, skip_special_tokens=False),
                                **metrics,
                            }
                        )

    metric_names = [
        "kl_p_to_q",
        "kl_q_to_p",
        "js",
        "tv",
        "accept_mass",
        "top1_match",
        f"top{args.topk}_overlap",
        "draft_prob_on_target_top1",
        "target_prob_on_target_top1",
        "gen_exact_match",
        "gen_prefix_match_len",
    ]
    summary_rows = aggregate_rows(raw_rows, metric_names=metric_names)

    config = {
        "args": vars(args),
        "num_layers": num_layers,
        "model": args.model,
    }

    write_json(config, os.path.join(args.out_dir, "config.json"))
    write_jsonl(raw_rows, os.path.join(args.out_dir, "raw_metrics.jsonl"))
    write_jsonl(example_rows, os.path.join(args.out_dir, "examples.jsonl"))
    write_csv(summary_rows, os.path.join(args.out_dir, "summary.csv"))
    write_json(summary_rows, os.path.join(args.out_dir, "summary.json"))

    print("\nSaved:")
    print(f"  {os.path.join(args.out_dir, 'summary.csv')}")
    print(f"  {os.path.join(args.out_dir, 'summary.json')}")
    print(f"  {os.path.join(args.out_dir, 'raw_metrics.jsonl')}")
    print(f"  {os.path.join(args.out_dir, 'examples.jsonl')}")


if __name__ == "__main__":
    main()
