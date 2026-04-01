#!/usr/bin/env python3
import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from kv_utils import (
    RidgeAccumulator,
    affine_apply,
    as_legacy_cache,
    depth_layer_map,
    distribution_metrics,
    flatten_kv,
    get_head_dim,
    get_kv_dim,
    get_num_kv_heads,
    iter_token_blocks,
    legacy_to_cache,
    load_causal_lm,
    load_tokenizer,
    set_seed,
    tokenizer_compatibility_report,
    unflatten_kv,
    write_json,
)


def mean_cosine_rows(x: torch.Tensor, y: torch.Tensor) -> float:
    return float(F.cosine_similarity(x.float(), y.float(), dim=-1).mean().item())


def write_csv(rows: List[Dict], path: str) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def solve_accumulators(accumulators: List[RidgeAccumulator]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    weights, biases = [], []
    for acc in accumulators:
        w, b = acc.solve()
        weights.append(w)
        biases.append(b)
    return weights, biases


def translate_legacy_cache(
    big_legacy_cache,
    layer_map: List[int],
    k_weights: List[torch.Tensor],
    k_biases: List[torch.Tensor],
    v_weights: List[torch.Tensor],
    v_biases: List[torch.Tensor],
    small_num_kv_heads: int,
    small_head_dim: int,
    out_device: str,
    out_dtype: torch.dtype,
):
    translated = []
    for small_layer_idx, big_layer_idx in enumerate(layer_map):
        k_big, v_big = big_legacy_cache[big_layer_idx]
        bsz, _, seqlen, _ = k_big.shape

        xk = flatten_kv(k_big).to(device=out_device, dtype=torch.float32)
        xv = flatten_kv(v_big).to(device=out_device, dtype=torch.float32)

        yk = affine_apply(xk, k_weights[small_layer_idx].to(out_device), k_biases[small_layer_idx].to(out_device))
        yv = affine_apply(xv, v_weights[small_layer_idx].to(out_device), v_biases[small_layer_idx].to(out_device))

        k_small = unflatten_kv(yk.to(dtype=out_dtype), bsz, seqlen, small_num_kv_heads, small_head_dim)
        v_small = unflatten_kv(yv.to(dtype=out_dtype), bsz, seqlen, small_num_kv_heads, small_head_dim)
        translated.append((k_small, v_small))
    return tuple(translated)


def identity_translate_legacy_cache(
    big_legacy_cache,
    layer_map: List[int],
    small_num_kv_heads: int,
    small_head_dim: int,
    out_device: str,
    out_dtype: torch.dtype,
):
    translated = []
    for _, big_layer_idx in enumerate(layer_map):
        k_big, v_big = big_legacy_cache[big_layer_idx]
        bsz, _, seqlen, _ = k_big.shape
        xk = flatten_kv(k_big).to(device=out_device)
        xv = flatten_kv(v_big).to(device=out_device)
        k_small = unflatten_kv(xk.to(dtype=out_dtype), bsz, seqlen, small_num_kv_heads, small_head_dim)
        v_small = unflatten_kv(xv.to(dtype=out_dtype), bsz, seqlen, small_num_kv_heads, small_head_dim)
        translated.append((k_small, v_small))
    return tuple(translated)


def merge_legacy_caches(
    native_small_legacy,
    translated_small_legacy,
    translated_k_layers: Optional[List[int]] = None,
    translated_v_layers: Optional[List[int]] = None,
):
    num_layers = len(native_small_legacy)
    if translated_k_layers is None:
        translated_k_layers = list(range(num_layers))
    if translated_v_layers is None:
        translated_v_layers = list(range(num_layers))

    translated_k_layers = set(translated_k_layers)
    translated_v_layers = set(translated_v_layers)

    merged = []
    for layer_idx in range(num_layers):
        k_native, v_native = native_small_legacy[layer_idx]
        k_trans, v_trans = translated_small_legacy[layer_idx]

        k_out = k_trans if layer_idx in translated_k_layers else k_native
        v_out = v_trans if layer_idx in translated_v_layers else v_native
        merged.append((k_out, v_out))
    return tuple(merged)


def metric_prefix_dict(prefix: str, ref_logits: torch.Tensor, test_logits: torch.Tensor, topk: int) -> Dict[str, float]:
    metrics = distribution_metrics(ref_logits, test_logits, topk=topk)
    out = {
        f"{prefix}_top1_match": float(
            (test_logits.argmax(dim=-1) == ref_logits.argmax(dim=-1)).float().mean().item()
        )
    }
    out.update({f"{prefix}_{k}": v for k, v in metrics.items()})
    return out


def aggregate_metric_rows(rows: List[Dict], exclude: Optional[List[str]] = None) -> Dict[str, float]:
    exclude = set(exclude or [])
    out = {}
    if not rows:
        return out
    keys = [k for k in rows[0].keys() if k not in exclude]
    for key in keys:
        vals = [float(r[key]) for r in rows if key in r]
        if vals:
            out[key] = float(sum(vals) / len(vals))
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit a layerwise linear KV-cache translator from a big model to a small model.")
    parser.add_argument("--big_model", type=str, default="Qwen/Qwen2.5-3B")
    parser.add_argument("--small_model", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--big_device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--small_device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--big_dtype", type=str, default="bf16")
    parser.add_argument("--small_dtype", type=str, default="bf16")
    parser.add_argument("--dataset_name", type=str, default="wikitext")
    parser.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--text_file", type=str, default=None)
    parser.add_argument("--text_column", type=str, default=None)
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--eval_split", type=str, default="validation")
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--train_sequences", type=int, default=512)
    parser.add_argument("--eval_sequences", type=int, default=128)
    parser.add_argument("--position_stride", type=int, default=1)
    parser.add_argument("--lambda_reg", type=float, default=1e-4)
    parser.add_argument("--layer_map", type=str, choices=["depth"], default="depth")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--shuffle_train", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow_incompatible_tokenizers", action="store_true")
    parser.add_argument("--out_dir", type=str, default="outputs/kv_linear_probe")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.seq_len < 2:
        raise ValueError("seq_len must be at least 2")
    if args.position_stride < 1:
        raise ValueError("position_stride must be >= 1")

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    print("Loading models and tokenizers...")
    big_tokenizer = load_tokenizer(args.big_model)
    small_tokenizer = load_tokenizer(args.small_model)
    compatibility = tokenizer_compatibility_report(big_tokenizer, small_tokenizer)
    if (not compatibility["all_probe_encodings_match"]) and (not args.allow_incompatible_tokenizers):
        raise ValueError(
            "Tokenizers appear incompatible. Use a same-tokenizer family or pass --allow_incompatible_tokenizers if you know what you're doing."
        )

    big_model = load_causal_lm(args.big_model, device=args.big_device, dtype_name=args.big_dtype)
    small_model = load_causal_lm(args.small_model, device=args.small_device, dtype_name=args.small_dtype)
    small_model_dtype = next(small_model.parameters()).dtype

    num_big_layers = int(big_model.config.num_hidden_layers)
    num_small_layers = int(small_model.config.num_hidden_layers)
    big_kv_dim = get_kv_dim(big_model.config)
    small_kv_dim = get_kv_dim(small_model.config)
    small_num_kv_heads = get_num_kv_heads(small_model.config)
    small_head_dim = get_head_dim(small_model.config)

    if args.layer_map == "depth":
        layer_map = depth_layer_map(num_big_layers, num_small_layers)
    else:
        raise ValueError(f"Unsupported layer_map: {args.layer_map}")

    print("Tokenizer compatibility:", compatibility)
    print(f"Big model:   {args.big_model}")
    print(f"Small model: {args.small_model}")
    print(f"Big layers={num_big_layers}, small layers={num_small_layers}")
    print(f"Big KV dim={big_kv_dim}, small KV dim={small_kv_dim}")
    print(f"Layer map: {layer_map}")

    k_accs = [RidgeAccumulator(big_kv_dim, small_kv_dim, lambda_reg=args.lambda_reg) for _ in range(num_small_layers)]
    v_accs = [RidgeAccumulator(big_kv_dim, small_kv_dim, lambda_reg=args.lambda_reg) for _ in range(num_small_layers)]

    print("\nCollecting train statistics for ridge fit...")
    train_iter = iter_token_blocks(
        tokenizer=big_tokenizer,
        seq_len=args.seq_len,
        max_blocks=args.train_sequences,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.train_split,
        text_file=args.text_file,
        text_column=args.text_column,
        shuffle=args.shuffle_train,
        seed=args.seed,
    )

    for train_idx, block in enumerate(train_iter):
        full_ids_big = block.unsqueeze(0).to(args.big_device)
        full_ids_small = block.unsqueeze(0).to(args.small_device)
        with torch.no_grad():
            big_out = big_model(input_ids=full_ids_big, use_cache=True)
            small_out = small_model(input_ids=full_ids_small, use_cache=True)
        big_legacy = as_legacy_cache(big_out.past_key_values)
        small_legacy = as_legacy_cache(small_out.past_key_values)

        for small_layer_idx, big_layer_idx in enumerate(layer_map):
            k_big, v_big = big_legacy[big_layer_idx]
            k_small, v_small = small_legacy[small_layer_idx]
            xk = flatten_kv(k_big).float()[:: args.position_stride]
            xv = flatten_kv(v_big).float()[:: args.position_stride]
            yk = flatten_kv(k_small).float()[:: args.position_stride]
            yv = flatten_kv(v_small).float()[:: args.position_stride]
            k_accs[small_layer_idx].update(xk, yk)
            v_accs[small_layer_idx].update(xv, yv)

        if train_idx % 20 == 0:
            print(f"  train sequence {train_idx + 1} / {args.train_sequences}")

    print("\nSolving ridge regressions...")
    k_weights, k_biases = solve_accumulators(k_accs)
    v_weights, v_biases = solve_accumulators(v_accs)

    translator_state = {
        "big_model": args.big_model,
        "small_model": args.small_model,
        "layer_map": layer_map,
        "big_kv_dim": big_kv_dim,
        "small_kv_dim": small_kv_dim,
        "small_num_kv_heads": small_num_kv_heads,
        "small_head_dim": small_head_dim,
        "lambda_reg": args.lambda_reg,
        "k_weights": [w.cpu() for w in k_weights],
        "k_biases": [b.cpu() for b in k_biases],
        "v_weights": [w.cpu() for w in v_weights],
        "v_biases": [b.cpu() for b in v_biases],
    }
    translator_path = os.path.join(args.out_dir, "translator.pt")
    torch.save(translator_state, translator_path)

    print("\nEvaluating reconstruction + next-token behavior...")
    recon_rows = []
    next_token_rows = []
    eval_iter = iter_token_blocks(
        tokenizer=big_tokenizer,
        seq_len=args.seq_len,
        max_blocks=args.eval_sequences,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.eval_split,
        text_file=args.text_file,
        text_column=args.text_column,
        shuffle=False,
        seed=args.seed,
    )

    identity_possible = big_kv_dim == small_kv_dim

    for eval_idx, block in enumerate(eval_iter):
        full_ids_big = block.unsqueeze(0).to(args.big_device)
        full_ids_small = block.unsqueeze(0).to(args.small_device)

        # Full-sequence caches for reconstruction metrics.
        with torch.no_grad():
            big_full_out = big_model(input_ids=full_ids_big, use_cache=True)
            small_full_out = small_model(input_ids=full_ids_small, use_cache=True)
        big_full_legacy = as_legacy_cache(big_full_out.past_key_values)
        small_full_legacy = as_legacy_cache(small_full_out.past_key_values)
        translated_full_legacy = translate_legacy_cache(
            big_legacy_cache=big_full_legacy,
            layer_map=layer_map,
            k_weights=k_weights,
            k_biases=k_biases,
            v_weights=v_weights,
            v_biases=v_biases,
            small_num_kv_heads=small_num_kv_heads,
            small_head_dim=small_head_dim,
            out_device=args.small_device,
            out_dtype=small_model_dtype,
        )
        identity_full_legacy = None
        if identity_possible:
            identity_full_legacy = identity_translate_legacy_cache(
                big_legacy_cache=big_full_legacy,
                layer_map=layer_map,
                small_num_kv_heads=small_num_kv_heads,
                small_head_dim=small_head_dim,
                out_device=args.small_device,
                out_dtype=small_model_dtype,
            )

        for layer_idx in range(num_small_layers):
            k_hat, v_hat = translated_full_legacy[layer_idx]
            k_ref, v_ref = small_full_legacy[layer_idx]
            xk = flatten_kv(k_hat)
            xv = flatten_kv(v_hat)
            yk = flatten_kv(k_ref)
            yv = flatten_kv(v_ref)
            row = {
                "eval_idx": eval_idx,
                "layer_idx": layer_idx,
                "big_layer_idx": layer_map[layer_idx],
                "k_mse": float(F.mse_loss(xk.float(), yk.float()).item()),
                "v_mse": float(F.mse_loss(xv.float(), yv.float()).item()),
                "k_cos": mean_cosine_rows(xk, yk),
                "v_cos": mean_cosine_rows(xv, yv),
            }
            if identity_possible:
                k_id, v_id = identity_full_legacy[layer_idx]
                xk_id = flatten_kv(k_id)
                xv_id = flatten_kv(v_id)
                row.update(
                    {
                        "k_mse_identity": float(F.mse_loss(xk_id.float(), yk.float()).item()),
                        "v_mse_identity": float(F.mse_loss(xv_id.float(), yv.float()).item()),
                        "k_cos_identity": mean_cosine_rows(xk_id, yk),
                        "v_cos_identity": mean_cosine_rows(xv_id, yv),
                    }
                )
            recon_rows.append(row)

        # Context-only cache test for the actual use case.
        context_big = full_ids_big[:, :-1]
        context_small = full_ids_small[:, :-1]
        current_big = full_ids_big[:, -1:]
        current_small = full_ids_small[:, -1:]

        with torch.no_grad():
            big_context_out = big_model(input_ids=context_big, use_cache=True)
            small_context_out = small_model(input_ids=context_small, use_cache=True)
            big_next_out = big_model(input_ids=current_big, past_key_values=big_context_out.past_key_values, use_cache=True)
            small_native_next_out = small_model(
                input_ids=current_small,
                past_key_values=small_context_out.past_key_values,
                use_cache=True,
            )

        big_context_legacy = as_legacy_cache(big_context_out.past_key_values)
        small_context_legacy = as_legacy_cache(small_context_out.past_key_values)

        translated_context_legacy = translate_legacy_cache(
            big_legacy_cache=big_context_legacy,
            layer_map=layer_map,
            k_weights=k_weights,
            k_biases=k_biases,
            v_weights=v_weights,
            v_biases=v_biases,
            small_num_kv_heads=small_num_kv_heads,
            small_head_dim=small_head_dim,
            out_device=args.small_device,
            out_dtype=small_model_dtype,
        )

        all_layers = list(range(num_small_layers))
        later_layers = list(range(1, num_small_layers))

        eval_caches = {
            "translated": translated_context_legacy,
            "konly": merge_legacy_caches(
                native_small_legacy=small_context_legacy,
                translated_small_legacy=translated_context_legacy,
                translated_k_layers=all_layers,
                translated_v_layers=[],
            ),
            "vonly": merge_legacy_caches(
                native_small_legacy=small_context_legacy,
                translated_small_legacy=translated_context_legacy,
                translated_k_layers=[],
                translated_v_layers=all_layers,
            ),
            "layer0k_native": merge_legacy_caches(
                native_small_legacy=small_context_legacy,
                translated_small_legacy=translated_context_legacy,
                translated_k_layers=later_layers,
                translated_v_layers=all_layers,
            ),
        }

        eval_outputs = {}
        for mode_name, mode_legacy in eval_caches.items():
            mode_cache = legacy_to_cache(mode_legacy)
            with torch.no_grad():
                eval_outputs[mode_name] = small_model(
                    input_ids=current_small,
                    past_key_values=mode_cache,
                    use_cache=True,
                )

        identity_metrics = {}
        if identity_possible:
            identity_context_legacy = identity_translate_legacy_cache(
                big_legacy_cache=big_context_legacy,
                layer_map=layer_map,
                small_num_kv_heads=small_num_kv_heads,
                small_head_dim=small_head_dim,
                out_device=args.small_device,
                out_dtype=small_model_dtype,
            )
            identity_context_cache = legacy_to_cache(identity_context_legacy)
            with torch.no_grad():
                small_identity_next_out = small_model(
                    input_ids=current_small,
                    past_key_values=identity_context_cache,
                    use_cache=True,
                )
            identity_logits = small_identity_next_out.logits[:, -1, :]
            identity_metrics = metric_prefix_dict(
                "identity_vs_big",
                big_next_out.logits[:, -1, :].to(identity_logits.device),
                identity_logits,
                topk=args.topk,
            )

        big_logits = big_next_out.logits[:, -1, :].to(args.small_device)
        small_native_logits = small_native_next_out.logits[:, -1, :]

        translated_logits = eval_outputs["translated"].logits[:, -1, :]
        konly_logits = eval_outputs["konly"].logits[:, -1, :]
        vonly_logits = eval_outputs["vonly"].logits[:, -1, :]
        layer0k_native_logits = eval_outputs["layer0k_native"].logits[:, -1, :]

        next_row = {
            "eval_idx": eval_idx,
            **metric_prefix_dict("native_vs_big", big_logits, small_native_logits, topk=args.topk),
            **metric_prefix_dict("translated_vs_big", big_logits, translated_logits, topk=args.topk),
            **metric_prefix_dict("translated_vs_native", small_native_logits, translated_logits, topk=args.topk),

            **metric_prefix_dict("konly_vs_big", big_logits, konly_logits, topk=args.topk),
            **metric_prefix_dict("konly_vs_native", small_native_logits, konly_logits, topk=args.topk),

            **metric_prefix_dict("vonly_vs_big", big_logits, vonly_logits, topk=args.topk),
            **metric_prefix_dict("vonly_vs_native", small_native_logits, vonly_logits, topk=args.topk),

            **metric_prefix_dict("layer0k_native_vs_big", big_logits, layer0k_native_logits, topk=args.topk),
            **metric_prefix_dict("layer0k_native_vs_native", small_native_logits, layer0k_native_logits, topk=args.topk),

            **identity_metrics,
        }
        next_token_rows.append(next_row)

        if eval_idx % 10 == 0:
            print(f"  eval sequence {eval_idx + 1} / {args.eval_sequences}")

    recon_summary = aggregate_metric_rows(recon_rows, exclude=["eval_idx", "layer_idx", "big_layer_idx"])
    next_summary = aggregate_metric_rows(next_token_rows, exclude=["eval_idx"])

    per_layer_summary = []
    layer_groups = defaultdict(list)
    for row in recon_rows:
        layer_groups[row["layer_idx"]].append(row)
    for layer_idx, rows in sorted(layer_groups.items()):
        summary = aggregate_metric_rows(rows, exclude=["eval_idx", "layer_idx", "big_layer_idx"])
        summary["layer_idx"] = layer_idx
        summary["big_layer_idx"] = layer_map[layer_idx]
        per_layer_summary.append(summary)

    result = {
        "args": vars(args),
        "tokenizer_compatibility": compatibility,
        "layer_map": layer_map,
        "big_model": args.big_model,
        "small_model": args.small_model,
        "big_num_layers": num_big_layers,
        "small_num_layers": num_small_layers,
        "big_kv_dim": big_kv_dim,
        "small_kv_dim": small_kv_dim,
        "reconstruction_summary": recon_summary,
        "next_token_summary": next_summary,
        "identity_possible": identity_possible,
        "translator_path": translator_path,
    }

    write_json(result, os.path.join(args.out_dir, "summary.json"))
    write_csv(recon_rows, os.path.join(args.out_dir, "reconstruction_rows.csv"))
    write_csv(per_layer_summary, os.path.join(args.out_dir, "reconstruction_per_layer.csv"))
    write_csv(next_token_rows, os.path.join(args.out_dir, "next_token_rows.csv"))

    print("\nSaved:")
    print(f"  {translator_path}")
    print(f"  {os.path.join(args.out_dir, 'summary.json')}")
    print(f"  {os.path.join(args.out_dir, 'reconstruction_rows.csv')}")
    print(f"  {os.path.join(args.out_dir, 'reconstruction_per_layer.csv')}")
    print(f"  {os.path.join(args.out_dir, 'next_token_rows.csv')}")

    print("\nHigh-level summary:")
    print(result)


if __name__ == "__main__":
    main()