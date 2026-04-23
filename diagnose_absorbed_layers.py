#!/usr/bin/env python3
"""
Layer-by-layer diagnostics for absorbed shared-cache draft forwards.

This script answers: "where does the absorbed model drift away from the native draft?"
It runs the native draft and absorbed draft side-by-side on the same prefix and logs per-layer
attention output, MLP output, residual-stream cosine similarity, L2 error, and RMS ratios.
"""

import argparse
import atexit
import csv
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from eval_absorbed_spec_decode import (
    affine_apply,
    apply_output_norm_match,
    apply_rotary_pos_emb_q_only,
    build_causal_mask,
    compute_native_attention_output,
    load_absorbed_state,
    parse_csv_items,
    parse_shared_layer_spec,
    repeat_kv,
)
from kv_utils import (
    as_legacy_cache,
    distribution_metrics,
    flatten_kv,
    iter_token_blocks,
    load_causal_lm,
    load_tokenizer,
    set_seed,
    tokenizer_compatibility_report,
    unflatten_kv,
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


def tensor_rms(tensor: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(tensor.float().square())).item())


def pair_stats(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    pred_flat = pred.float().reshape(-1, pred.shape[-1])
    target_flat = target.float().reshape(-1, target.shape[-1])
    pred_rms = tensor_rms(pred_flat)
    target_rms = tensor_rms(target_flat)
    return {
        "cosine": float(F.cosine_similarity(pred_flat, target_flat, dim=-1).mean().item()),
        "l2": float(torch.norm(pred_flat - target_flat, p=2, dim=-1).mean().item()),
        "max_abs": float((pred_flat - target_flat).abs().max().item()),
        "pred_rms": pred_rms,
        "target_rms": target_rms,
        "rms_ratio": float(pred_rms / max(target_rms, 1e-12)),
    }


def aggregate_by_layer(rows: List[Dict[str, Any]], num_layers: int) -> List[Dict[str, float]]:
    out: List[Dict[str, float]] = []
    for layer_idx in range(num_layers):
        layer_rows = [row for row in rows if int(row["layer_idx"]) == layer_idx]
        if not layer_rows:
            continue
        numeric_keys = [key for key in layer_rows[0].keys() if key not in {"prompt_idx", "layer_idx", "shared_layer"}]
        summary: Dict[str, float] = {"layer_idx": float(layer_idx), "num_prompts": float(len(layer_rows))}
        for key in numeric_keys:
            vals = [float(row[key]) for row in layer_rows if key in row]
            if vals:
                summary[key] = float(sum(vals) / len(vals))
        summary["shared_layer"] = float(layer_rows[0]["shared_layer"])
        out.append(summary)
    return out


def compute_absorbed_attention_output(
    *,
    layer,
    hidden_states_ln: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor,
    big_legacy_cache,
    absorbed_state: Dict[str, Any],
    small_layer_idx: int,
    shared_variant: str,
    norm_match: str,
) -> torch.Tensor:
    bsz, seq_len, _ = hidden_states_ln.shape
    config = getattr(layer.self_attn, "config", None)
    small_q_heads = absorbed_state.get("small_q_heads")
    if small_q_heads is None and config is not None:
        small_q_heads = config.num_attention_heads
    if small_q_heads is None:
        small_q_heads = getattr(layer.self_attn, "num_heads")
    small_q_heads = int(small_q_heads)

    small_head_dim = absorbed_state.get("small_head_dim")
    if small_head_dim is None and config is not None:
        small_head_dim = getattr(config, "head_dim", None) or (config.hidden_size // small_q_heads)
    if small_head_dim is None:
        small_head_dim = getattr(layer.self_attn, "head_dim")
    small_head_dim = int(small_head_dim)

    small_kv_heads = absorbed_state.get("small_kv_heads")
    if small_kv_heads is None:
        small_kv_heads = layer.self_attn.k_proj.out_features // small_head_dim
    small_kv_heads = int(small_kv_heads)
    target_layer_idx = int(absorbed_state["layer_map"][small_layer_idx])
    k_big, v_big = big_legacy_cache[target_layer_idx]
    k_big = k_big.to(hidden_states_ln.device)
    v_big = v_big.to(hidden_states_ln.device)

    q = layer.self_attn.q_proj(hidden_states_ln).view(bsz, seq_len, small_q_heads, small_head_dim).transpose(1, 2)
    q = apply_rotary_pos_emb_q_only(q, *position_embeddings)

    k_weight = absorbed_state["k_weights"][small_layer_idx].to(hidden_states_ln.device)
    k_bias = absorbed_state["k_biases"][small_layer_idx].to(hidden_states_ln.device)
    k_shared_flat = affine_apply(flatten_kv(k_big), k_weight, k_bias)
    k_shared = unflatten_kv(k_shared_flat, bsz, seq_len, small_kv_heads, small_head_dim)
    if small_q_heads % small_kv_heads != 0:
        raise ValueError(f"small_q_heads={small_q_heads} is not divisible by small_kv_heads={small_kv_heads}")
    shared_k = repeat_kv(k_shared, small_q_heads // small_kv_heads)

    scaling = float(getattr(layer.self_attn, "scaling", small_head_dim ** -0.5))
    attn_scores = torch.matmul(q.float(), shared_k.transpose(2, 3).float()) * scaling
    attn_scores = attn_scores + attention_mask
    attn_weights = torch.softmax(attn_scores, dim=-1, dtype=torch.float32).to(q.dtype)

    if shared_variant == "full":
        target_kv_heads = v_big.shape[1]
        if small_q_heads % target_kv_heads != 0:
            raise ValueError(f"small_q_heads={small_q_heads} is not divisible by target_kv_heads={target_kv_heads}")
        shared_v = repeat_kv(v_big, small_q_heads // target_kv_heads)
        h_tilde = torch.matmul(attn_weights.float(), shared_v.float())
        h_tilde = h_tilde.transpose(1, 2).reshape(bsz * seq_len, small_q_heads * small_head_dim)
        o_weight = absorbed_state["o_weights"][small_layer_idx].to(hidden_states_ln.device)
        o_bias = absorbed_state["o_biases"][small_layer_idx].to(hidden_states_ln.device)
        attn_output = affine_apply(h_tilde, o_weight, o_bias).view(bsz, seq_len, -1).to(hidden_states_ln.dtype)
        return apply_output_norm_match(
            attn_output,
            absorbed_state=absorbed_state,
            layer_idx=small_layer_idx,
            norm_match=norm_match,
        )
    if shared_variant == "k_only":
        native_kv_heads = int(layer.self_attn.v_proj.out_features // small_head_dim)
        v_native = layer.self_attn.v_proj(hidden_states_ln).view(
            bsz, seq_len, native_kv_heads, small_head_dim
        ).transpose(1, 2)
        if small_q_heads % native_kv_heads != 0:
            raise ValueError(f"small_q_heads={small_q_heads} is not divisible by native_kv_heads={native_kv_heads}")
        v_native = repeat_kv(v_native, small_q_heads // native_kv_heads)
        attn_heads = torch.matmul(attn_weights.float(), v_native.float())
        attn_heads = attn_heads.transpose(1, 2).reshape(bsz, seq_len, small_q_heads * small_head_dim)
        return layer.self_attn.o_proj(attn_heads.to(hidden_states_ln.dtype))
    raise ValueError(f"Unsupported shared_variant: {shared_variant}")


@torch.no_grad()
def trace_prompt(
    *,
    small_model,
    input_ids: torch.Tensor,
    big_legacy_cache,
    absorbed_state: Dict[str, Any],
    small_device: str,
    shared_layer_indices: Sequence[int],
    shared_variant: str,
    norm_match: str,
    topk: int,
) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
    model = small_model.model
    native_hidden = model.embed_tokens(input_ids.to(small_device))
    absorbed_hidden = native_hidden.clone()
    bsz, seq_len, _ = native_hidden.shape
    if bsz != 1:
        raise ValueError("This diagnostic script currently expects batch size 1.")

    position_ids = torch.arange(seq_len, device=native_hidden.device, dtype=torch.long).unsqueeze(0)
    position_embeddings = model.rotary_emb(native_hidden, position_ids)
    attention_mask = build_causal_mask(seq_len, native_hidden.device)
    shared_layer_set = set(shared_layer_indices)

    layer_rows: List[Dict[str, float]] = []
    for layer_idx, layer in enumerate(model.layers):
        native_residual = native_hidden
        native_ln = layer.input_layernorm(native_hidden)
        native_attn = compute_native_attention_output(
            attn_module=layer.self_attn,
            hidden_states=native_ln,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        ).to(native_hidden.dtype)
        native_after_attn = native_residual + native_attn
        native_mlp_in = layer.post_attention_layernorm(native_after_attn)
        native_mlp = layer.mlp(native_mlp_in)
        native_after_layer = native_after_attn + native_mlp

        absorbed_residual = absorbed_hidden
        absorbed_ln = layer.input_layernorm(absorbed_hidden)
        if layer_idx in shared_layer_set:
            absorbed_attn = compute_absorbed_attention_output(
                layer=layer,
                hidden_states_ln=absorbed_ln,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                big_legacy_cache=big_legacy_cache,
                absorbed_state=absorbed_state,
                small_layer_idx=layer_idx,
                shared_variant=shared_variant,
                norm_match=norm_match,
            ).to(absorbed_hidden.dtype)
        else:
            absorbed_attn = compute_native_attention_output(
                attn_module=layer.self_attn,
                hidden_states=absorbed_ln,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
            ).to(absorbed_hidden.dtype)
        absorbed_after_attn = absorbed_residual + absorbed_attn
        absorbed_mlp_in = layer.post_attention_layernorm(absorbed_after_attn)
        absorbed_mlp = layer.mlp(absorbed_mlp_in)
        absorbed_after_layer = absorbed_after_attn + absorbed_mlp

        row: Dict[str, float] = {
            "layer_idx": float(layer_idx),
            "shared_layer": float(layer_idx in shared_layer_set),
        }
        for prefix, stats in {
            "attn": pair_stats(absorbed_attn, native_attn),
            "after_attn": pair_stats(absorbed_after_attn, native_after_attn),
            "mlp": pair_stats(absorbed_mlp, native_mlp),
            "after_layer": pair_stats(absorbed_after_layer, native_after_layer),
        }.items():
            for key, value in stats.items():
                row[f"{prefix}_{key}"] = value
        layer_rows.append(row)

        native_hidden = native_after_layer
        absorbed_hidden = absorbed_after_layer

    native_logits = small_model.lm_head(model.norm(native_hidden)[:, -1:, :])[:, -1, :]
    absorbed_logits = small_model.lm_head(model.norm(absorbed_hidden)[:, -1:, :])[:, -1, :]
    final_metrics = distribution_metrics(native_logits, absorbed_logits, topk=topk)
    final_metrics["native_vs_absorbed_top1_match"] = float(
        (native_logits.argmax(dim=-1) == absorbed_logits.argmax(dim=-1)).float().mean().item()
    )
    return layer_rows, final_metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose layer-wise drift in absorbed shared-cache draft forwards.")
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
    parser.add_argument("--num_prompts", type=int, default=64)
    parser.add_argument("--topk", type=int, default=5)
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
    )
    parser.add_argument(
        "--norm_match",
        type=str,
        choices=["none", "rms", "std"],
        default="none",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow_incompatible_tokenizers", action="store_true")
    parser.add_argument("--out_dir", type=str, default="outputs/absorbed_layer_diagnostics")
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
    print(f"Shared layers ({len(shared_layer_indices)}/{num_layers}): {shared_layer_indices}")
    print(f"Shared variant: {args.shared_variant}")
    print(f"Output norm matching: {args.norm_match}")

    prompt_iter = iter_token_blocks(
        tokenizer=big_tokenizer,
        seq_len=args.prompt_len,
        max_blocks=args.num_prompts,
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

    all_layer_rows: List[Dict[str, Any]] = []
    final_rows: List[Dict[str, Any]] = []
    for prompt_idx, block in enumerate(prompt_iter):
        prompt_ids = block.unsqueeze(0)
        with torch.no_grad():
            big_out = big_model(input_ids=prompt_ids.to(args.big_device), use_cache=True)
        big_legacy_cache = as_legacy_cache(big_out.past_key_values)
        layer_rows, final_metrics = trace_prompt(
            small_model=small_model,
            input_ids=prompt_ids,
            big_legacy_cache=big_legacy_cache,
            absorbed_state=absorbed_state,
            small_device=args.small_device,
            shared_layer_indices=shared_layer_indices,
            shared_variant=args.shared_variant,
            norm_match=args.norm_match,
            topk=args.topk,
        )
        for row in layer_rows:
            row["prompt_idx"] = int(prompt_idx)
            all_layer_rows.append(row)
        final_row = {"prompt_idx": int(prompt_idx), **final_metrics}
        final_rows.append(final_row)

        if wandb_run is not None:
            log_payload: Dict[str, float] = {}
            for row in layer_rows:
                layer_idx = int(row["layer_idx"])
                for key, value in row.items():
                    if key in {"prompt_idx", "layer_idx", "shared_layer"}:
                        continue
                    log_payload[f"diagnostic/{key}/layer_{layer_idx}"] = float(value)
                log_payload[f"diagnostic/shared_layer/layer_{layer_idx}"] = float(row["shared_layer"])
            for key, value in final_metrics.items():
                log_payload[f"diagnostic/final_{key}"] = float(value)
            wandb_run.log(log_payload, step=prompt_idx + 1)

        if (prompt_idx + 1) % 10 == 0:
            print(f"Processed {prompt_idx + 1} / {args.num_prompts} prompts")

    layer_summary = aggregate_by_layer(all_layer_rows, num_layers)
    final_summary: Dict[str, float] = {}
    if final_rows:
        for key in final_rows[0].keys():
            if key == "prompt_idx":
                continue
            vals = [float(row[key]) for row in final_rows if key in row]
            final_summary[key] = float(sum(vals) / len(vals))

    write_csv(all_layer_rows, os.path.join(args.out_dir, "layer_rows.csv"))
    write_csv(layer_summary, os.path.join(args.out_dir, "layer_summary.csv"))
    write_csv(final_rows, os.path.join(args.out_dir, "final_rows.csv"))
    write_json(
        {
            "num_prompts": len(final_rows),
            "translator_path": args.translator_path,
            "translator_output_routing_source": absorbed_state.get("output_routing_source", "unknown"),
            "shared_layers_spec": args.shared_layers,
            "shared_layer_indices": shared_layer_indices,
            "shared_variant": args.shared_variant,
            "norm_match": args.norm_match,
            "final_summary": final_summary,
            "layer_summary": layer_summary,
        },
        os.path.join(args.out_dir, "summary.json"),
    )

    if wandb_run is not None:
        for key, value in final_summary.items():
            wandb_run.summary[f"diagnostic/final/{key}"] = value
        for row in layer_summary:
            layer_idx = int(row["layer_idx"])
            for key, value in row.items():
                if key in {"layer_idx"}:
                    continue
                wandb_run.summary[f"diagnostic/layer_summary/{key}/layer_{layer_idx}"] = value

    print("Done!")
    print(f"  {os.path.join(args.out_dir, 'layer_rows.csv')}")
    print(f"  {os.path.join(args.out_dir, 'layer_summary.csv')}")
    print(f"  {os.path.join(args.out_dir, 'final_rows.csv')}")
    print(f"  {os.path.join(args.out_dir, 'summary.json')}")


if __name__ == "__main__":
    main()
