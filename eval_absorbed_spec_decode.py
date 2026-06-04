#!/usr/bin/env python3
import argparse
import atexit
import csv
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from kv_utils import (
    as_legacy_cache,
    distribution_metrics,
    flatten_kv,
    get_head_dim,
    get_num_kv_heads,
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
    fieldnames = list(rows[0].keys())
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


def parse_csv_items(value: Optional[str]) -> List[str]:
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def aggregate_rows(rows: List[Dict[str, float]], exclude: Optional[Sequence[str]] = None) -> Dict[str, float]:
    exclude = set(exclude or [])
    out: Dict[str, float] = {}
    if not rows:
        return out
    keys = [k for k in rows[0].keys() if k not in exclude]
    for key in keys:
        vals = [float(row[key]) for row in rows if key in row]
        if vals:
            out[key] = float(sum(vals) / len(vals))
    return out


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_pair(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1).to(dtype=q.dtype, device=q.device)
    sin = sin.unsqueeze(1).to(dtype=q.dtype, device=q.device)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def apply_rotary_pos_emb_q_only(q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = cos.unsqueeze(1).to(dtype=q.dtype, device=q.device)
    sin = sin.unsqueeze(1).to(dtype=q.dtype, device=q.device)
    return (q * cos) + (rotate_half(q) * sin)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_key_value_heads, seqlen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, seqlen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, seqlen, head_dim)


def affine_apply(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return F.linear(x.float(), weight.float(), bias.float())


def _get_layer_scalar(absorbed_state: Dict[str, Any], key: str, layer_idx: int) -> float:
    values = absorbed_state.get(key)
    if values is None:
        raise ValueError(
            f"Translator checkpoint is missing {key}; re-run fit_kv_absorbed.py with the latest code "
            "before using --norm_match."
        )
    if layer_idx >= len(values):
        raise ValueError(f"Translator checkpoint key {key} has no entry for layer {layer_idx}.")
    return float(values[layer_idx])


def apply_output_norm_match(
    attn_output: torch.Tensor,
    *,
    absorbed_state: Dict[str, Any],
    layer_idx: int,
    norm_match: str,
    eps: float = 1e-6,
) -> torch.Tensor:
    if norm_match == "none":
        return attn_output

    dtype = attn_output.dtype
    x = attn_output.float()
    if norm_match == "rms":
        target_rms = _get_layer_scalar(absorbed_state, "o_target_rms", layer_idx)
        current_rms = torch.sqrt(torch.mean(x.square(), dim=-1, keepdim=True)).clamp_min(eps)
        return (x * (target_rms / current_rms)).to(dtype)
    if norm_match == "std":
        target_mean = _get_layer_scalar(absorbed_state, "o_target_mean", layer_idx)
        target_std = _get_layer_scalar(absorbed_state, "o_target_std", layer_idx)
        current_mean = x.mean(dim=-1, keepdim=True)
        current_std = x.std(dim=-1, keepdim=True, unbiased=False).clamp_min(eps)
        return ((x - current_mean) * (target_std / current_std) + target_mean).to(dtype)
    raise ValueError(f"Unsupported norm_match: {norm_match}")


def build_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    mask = torch.full((seq_len, seq_len), torch.finfo(torch.float32).min, device=device, dtype=torch.float32)
    mask = torch.triu(mask, diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)


def load_absorbed_state(path: str) -> Dict[str, Any]:
    state = torch.load(path, map_location="cpu")
    required = ["layer_map", "k_weights", "k_biases", "o_weights", "o_biases"]
    for key in required:
        if key not in state:
            raise ValueError(f"Absorbed translator checkpoint is missing required key: {key}")
    return state


def parse_shared_layer_spec(spec: str, num_layers: int) -> List[int]:
    spec = spec.strip().lower()
    if spec == "all":
        return list(range(num_layers))
    if spec == "none":
        return []
    if spec in {"every_other", "even"}:
        return list(range(0, num_layers, 2))
    if spec == "odd":
        return list(range(1, num_layers, 2))
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

    out = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        idx = int(item)
        if idx < 0 or idx >= num_layers:
            raise ValueError(f"Layer index {idx} is out of range for num_layers={num_layers}")
        out.append(idx)
    if not out:
        raise ValueError(f"Unsupported shared layer spec: {spec}")
    return sorted(set(out))


def compute_native_attention_output(
    *,
    attn_module,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    bsz, seq_len, _ = hidden_states.shape
    # Newer transformers moved num_heads/head_dim off Qwen2Attention onto config.
    config = getattr(attn_module, "config", None)
    num_heads = getattr(attn_module, "num_heads", None)
    if num_heads is None and config is not None:
        num_heads = config.num_attention_heads
    head_dim = getattr(attn_module, "head_dim", None)
    if head_dim is None and config is not None:
        head_dim = getattr(config, "head_dim", None) or (config.hidden_size // num_heads)
    num_heads = int(num_heads)
    head_dim = int(head_dim)
    num_kv_heads = int(attn_module.k_proj.out_features // head_dim)

    q = attn_module.q_proj(hidden_states).view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
    k = attn_module.k_proj(hidden_states).view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)
    v = attn_module.v_proj(hidden_states).view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)
    q, k = apply_rotary_pos_emb_pair(q, k, *position_embeddings)

    if num_heads % num_kv_heads != 0:
        raise ValueError(f"num_heads={num_heads} is not divisible by num_kv_heads={num_kv_heads}")
    k = repeat_kv(k, num_heads // num_kv_heads)
    v = repeat_kv(v, num_heads // num_kv_heads)

    scaling = float(getattr(attn_module, "scaling", head_dim ** -0.5))
    attn_scores = torch.matmul(q.float(), k.transpose(2, 3).float()) * scaling
    attn_scores = attn_scores + attention_mask
    attn_weights = torch.softmax(attn_scores, dim=-1, dtype=torch.float32).to(q.dtype)
    attn_output = torch.matmul(attn_weights.float(), v.float())
    attn_output = attn_output.transpose(1, 2).reshape(bsz, seq_len, num_heads * head_dim)
    return attn_module.o_proj(attn_output.to(hidden_states.dtype))


@torch.no_grad()
def absorbed_draft_logits_from_prefix_cache(
    *,
    small_model,
    input_ids: torch.Tensor,
    target_prefix_legacy_cache,
    target_prefix_len: int,
    absorbed_state: Dict[str, Any],
    small_device: str,
    shared_layer_indices: Optional[Sequence[int]] = None,
    shared_variant: str = "full",
    norm_match: str = "none",
) -> torch.Tensor:
    if shared_variant != "full":
        raise ValueError("prefix-cache absorbed decoding currently supports --shared_variant full only.")

    model = small_model.model
    hidden_states = model.embed_tokens(input_ids.to(small_device))
    bsz, seq_len, _ = hidden_states.shape
    if bsz != 1:
        raise ValueError("This integration script currently expects batch size 1.")
    if target_prefix_len < 1 or target_prefix_len > seq_len:
        raise ValueError(f"target_prefix_len={target_prefix_len} is invalid for seq_len={seq_len}.")

    position_ids = torch.arange(seq_len, device=hidden_states.device, dtype=torch.long).unsqueeze(0)
    position_embeddings = model.rotary_emb(hidden_states, position_ids)
    attention_mask = build_causal_mask(seq_len, hidden_states.device)

    small_q_heads = int(absorbed_state.get("small_q_heads", small_model.config.num_attention_heads))
    small_kv_heads = int(absorbed_state.get("small_kv_heads", get_num_kv_heads(small_model.config)))
    small_head_dim = int(absorbed_state.get("small_head_dim", get_head_dim(small_model.config)))
    layer_map = [int(x) for x in absorbed_state["layer_map"]]
    shared_layer_set = set(shared_layer_indices if shared_layer_indices is not None else range(len(layer_map)))

    for small_layer_idx, layer in enumerate(model.layers):
        residual = hidden_states
        hidden_states_ln = layer.input_layernorm(hidden_states)
        if small_layer_idx in shared_layer_set:
            target_layer_idx = layer_map[small_layer_idx]
            k_big, v_big = target_prefix_legacy_cache[target_layer_idx]
            k_big = k_big[:, :, :target_prefix_len, :].to(hidden_states.device)
            v_big = v_big[:, :, :target_prefix_len, :].to(hidden_states.device)
            if k_big.shape[2] != target_prefix_len or v_big.shape[2] != target_prefix_len:
                raise ValueError("Target prefix cache is shorter than target_prefix_len.")

            q_all = layer.self_attn.q_proj(hidden_states_ln).view(
                bsz, seq_len, small_q_heads, small_head_dim
            ).transpose(1, 2)
            k_native_all = layer.self_attn.k_proj(hidden_states_ln).view(
                bsz, seq_len, small_kv_heads, small_head_dim
            ).transpose(1, 2)
            q_all, k_native_all = apply_rotary_pos_emb_pair(q_all, k_native_all, *position_embeddings)

            if small_q_heads % small_kv_heads != 0:
                raise ValueError(
                    f"small_q_heads={small_q_heads} is not divisible by small_kv_heads={small_kv_heads}"
                )
            repeats = small_q_heads // small_kv_heads

            # Key absorption for the accepted prefix:
            #   q @ (K_target W_k^T + b_k)^T
            # = (q W_k) @ K_target^T + q @ b_k
            # This avoids materializing mapped prefix keys and keeps the draft cache tail-only.
            k_weight = absorbed_state["k_weights"][small_layer_idx].to(hidden_states.device)
            k_bias = absorbed_state["k_biases"][small_layer_idx].to(hidden_states.device)
            big_kv_dim = int(k_weight.shape[1])
            k_weight_by_q_head = k_weight.view(small_kv_heads, small_head_dim, big_kv_dim)
            k_weight_by_q_head = k_weight_by_q_head.repeat_interleave(repeats, dim=0)
            k_bias_by_q_head = k_bias.view(small_kv_heads, small_head_dim).repeat_interleave(repeats, dim=0)
            q_prefix = torch.einsum("bhtd,hdf->bhtf", q_all.float(), k_weight_by_q_head.float())
            k_big_flat = k_big.permute(0, 2, 1, 3).reshape(bsz, target_prefix_len, big_kv_dim).float()
            prefix_scores = torch.einsum("bhtf,bpf->bhtp", q_prefix, k_big_flat)
            prefix_scores = prefix_scores + torch.einsum(
                "bhtd,hd->bht",
                q_all.float(),
                k_bias_by_q_head.float(),
            ).unsqueeze(-1)

            k_tail = k_native_all[:, :, target_prefix_len:, :]
            tail_k = repeat_kv(k_tail, repeats)

            target_kv_heads = v_big.shape[1]
            target_head_dim = v_big.shape[-1]
            if target_head_dim != small_head_dim:
                raise ValueError(
                    f"Target head dim {target_head_dim} differs from draft head dim {small_head_dim}; "
                    "prefix-cache mode needs a value latent adapter for this pair."
                )
            native_kv_heads = int(layer.self_attn.v_proj.out_features // small_head_dim)
            v_tail = layer.self_attn.v_proj(hidden_states_ln[:, target_prefix_len:, :]).view(
                bsz,
                seq_len - target_prefix_len,
                native_kv_heads,
                small_head_dim,
            ).transpose(1, 2)
            if small_q_heads % target_kv_heads != 0:
                raise ValueError(
                    f"small_q_heads={small_q_heads} is not divisible by target_kv_heads={target_kv_heads}"
                )
            if small_q_heads % native_kv_heads != 0:
                raise ValueError(
                    f"small_q_heads={small_q_heads} is not divisible by native_kv_heads={native_kv_heads}"
                )
            shared_v = torch.cat(
                [
                    repeat_kv(v_big, small_q_heads // target_kv_heads),
                    repeat_kv(v_tail, small_q_heads // native_kv_heads),
                ],
                dim=2,
            )

            scaling = float(getattr(layer.self_attn, "scaling", small_head_dim ** -0.5))
            if k_tail.shape[2] == 0:
                tail_scores = prefix_scores.new_empty(bsz, small_q_heads, seq_len, 0)
            else:
                tail_scores = torch.matmul(q_all.float(), tail_k.transpose(2, 3).float())
            attn_scores = torch.cat([prefix_scores, tail_scores], dim=-1) * scaling
            attn_scores = attn_scores + attention_mask
            attn_weights = torch.softmax(attn_scores, dim=-1, dtype=torch.float32).to(q_all.dtype)
            h_tilde = torch.matmul(attn_weights.float(), shared_v.float())
            h_tilde = h_tilde.transpose(1, 2).reshape(bsz * seq_len, small_q_heads * small_head_dim)

            o_weight = absorbed_state["o_weights"][small_layer_idx].to(hidden_states.device)
            o_bias = absorbed_state["o_biases"][small_layer_idx].to(hidden_states.device)
            attn_output = affine_apply(h_tilde, o_weight, o_bias).view(bsz, seq_len, -1).to(hidden_states.dtype)
            attn_output = apply_output_norm_match(
                attn_output,
                absorbed_state=absorbed_state,
                layer_idx=small_layer_idx,
                norm_match=norm_match,
            )
        else:
            attn_output = compute_native_attention_output(
                attn_module=layer.self_attn,
                hidden_states=hidden_states_ln,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
            ).to(hidden_states.dtype)

        hidden_states = residual + attn_output
        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        hidden_states = layer.mlp(hidden_states)
        hidden_states = residual + hidden_states

    hidden_states = model.norm(hidden_states)
    logits = small_model.lm_head(hidden_states[:, -1:, :])
    return logits[:, -1, :]


@torch.no_grad()
def absorbed_draft_logits_from_target_cache(
    *,
    small_model,
    input_ids: torch.Tensor,
    big_legacy_cache,
    absorbed_state: Dict[str, Any],
    small_device: str,
    shared_layer_indices: Optional[Sequence[int]] = None,
    shared_variant: str = "full",
    norm_match: str = "none",
) -> torch.Tensor:
    model = small_model.model
    hidden_states = model.embed_tokens(input_ids.to(small_device))
    bsz, seq_len, _ = hidden_states.shape
    if bsz != 1:
        raise ValueError("This integration script currently expects batch size 1.")

    position_ids = torch.arange(seq_len, device=hidden_states.device, dtype=torch.long).unsqueeze(0)
    position_embeddings = model.rotary_emb(hidden_states, position_ids)
    attention_mask = build_causal_mask(seq_len, hidden_states.device)

    small_q_heads = int(absorbed_state.get("small_q_heads", small_model.config.num_attention_heads))
    small_kv_heads = int(absorbed_state.get("small_kv_heads", get_num_kv_heads(small_model.config)))
    small_head_dim = int(absorbed_state.get("small_head_dim", get_head_dim(small_model.config)))
    layer_map = [int(x) for x in absorbed_state["layer_map"]]
    shared_layer_set = set(shared_layer_indices if shared_layer_indices is not None else range(len(layer_map)))

    for small_layer_idx, layer in enumerate(model.layers):
        residual = hidden_states
        hidden_states_ln = layer.input_layernorm(hidden_states)
        if small_layer_idx in shared_layer_set:
            target_layer_idx = layer_map[small_layer_idx]
            k_big, v_big = big_legacy_cache[target_layer_idx]
            k_big = k_big.to(hidden_states.device)
            v_big = v_big.to(hidden_states.device)

            q = layer.self_attn.q_proj(hidden_states_ln).view(bsz, seq_len, small_q_heads, small_head_dim).transpose(1, 2)
            q = apply_rotary_pos_emb_q_only(q, *position_embeddings)

            k_weight = absorbed_state["k_weights"][small_layer_idx].to(hidden_states.device)
            k_bias = absorbed_state["k_biases"][small_layer_idx].to(hidden_states.device)
            k_shared_flat = affine_apply(flatten_kv(k_big), k_weight, k_bias)
            k_shared = unflatten_kv(k_shared_flat, bsz, seq_len, small_kv_heads, small_head_dim)

            if small_q_heads % small_kv_heads != 0:
                raise ValueError(
                    f"small_q_heads={small_q_heads} is not divisible by small_kv_heads={small_kv_heads}"
                )
            shared_k = repeat_kv(k_shared, small_q_heads // small_kv_heads)

            scaling = float(getattr(layer.self_attn, "scaling", small_head_dim ** -0.5))
            attn_scores = torch.matmul(q.float(), shared_k.transpose(2, 3).float()) * scaling
            attn_scores = attn_scores + attention_mask
            attn_weights = torch.softmax(attn_scores, dim=-1, dtype=torch.float32).to(q.dtype)

            if shared_variant == "full":
                target_kv_heads = v_big.shape[1]
                if small_q_heads % target_kv_heads != 0:
                    raise ValueError(
                        f"small_q_heads={small_q_heads} is not divisible by target_kv_heads={target_kv_heads}"
                    )
                shared_v = repeat_kv(v_big, small_q_heads // target_kv_heads)
                h_tilde = torch.matmul(attn_weights.float(), shared_v.float())
                h_tilde = h_tilde.transpose(1, 2).reshape(bsz * seq_len, small_q_heads * small_head_dim)

                o_weight = absorbed_state["o_weights"][small_layer_idx].to(hidden_states.device)
                o_bias = absorbed_state["o_biases"][small_layer_idx].to(hidden_states.device)
                attn_output = affine_apply(h_tilde, o_weight, o_bias).view(bsz, seq_len, -1).to(hidden_states.dtype)
                attn_output = apply_output_norm_match(
                    attn_output,
                    absorbed_state=absorbed_state,
                    layer_idx=small_layer_idx,
                    norm_match=norm_match,
                )
            elif shared_variant == "k_only":
                native_kv_heads = int(layer.self_attn.v_proj.out_features // small_head_dim)
                v_native = layer.self_attn.v_proj(hidden_states_ln).view(
                    bsz, seq_len, native_kv_heads, small_head_dim
                ).transpose(1, 2)
                if small_q_heads % native_kv_heads != 0:
                    raise ValueError(
                        f"small_q_heads={small_q_heads} is not divisible by native_kv_heads={native_kv_heads}"
                    )
                v_native = repeat_kv(v_native, small_q_heads // native_kv_heads)
                attn_heads = torch.matmul(attn_weights.float(), v_native.float())
                attn_heads = attn_heads.transpose(1, 2).reshape(bsz, seq_len, small_q_heads * small_head_dim)
                attn_output = layer.self_attn.o_proj(attn_heads.to(hidden_states.dtype))
            else:
                raise ValueError(f"Unsupported shared_variant: {shared_variant}")
        else:
            attn_output = compute_native_attention_output(
                attn_module=layer.self_attn,
                hidden_states=hidden_states_ln,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
            ).to(hidden_states.dtype)

        hidden_states = residual + attn_output
        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        hidden_states = layer.mlp(hidden_states)
        hidden_states = residual + hidden_states

    hidden_states = model.norm(hidden_states)
    logits = small_model.lm_head(hidden_states[:, -1:, :])
    return logits[:, -1, :]


@torch.no_grad()
def target_next_logits(big_model, prefix_ids: torch.Tensor, big_device: str) -> torch.Tensor:
    out = big_model(input_ids=prefix_ids.to(big_device), use_cache=False)
    return out.logits[:, -1, :]


@torch.no_grad()
def native_draft_next_logits(small_model, prefix_ids: torch.Tensor, small_device: str) -> torch.Tensor:
    out = small_model(input_ids=prefix_ids.to(small_device), use_cache=False)
    return out.logits[:, -1, :]


@torch.no_grad()
def absorbed_draft_next_logits(
    *,
    big_model,
    small_model,
    prefix_ids: torch.Tensor,
    absorbed_state: Dict[str, Any],
    big_device: str,
    small_device: str,
    shared_layer_indices: Optional[Sequence[int]] = None,
    shared_variant: str = "full",
    norm_match: str = "none",
) -> torch.Tensor:
    if shared_layer_indices is not None and len(shared_layer_indices) == 0:
        return native_draft_next_logits(small_model, prefix_ids, small_device)
    big_out = big_model(input_ids=prefix_ids.to(big_device), use_cache=True)
    big_legacy_cache = as_legacy_cache(big_out.past_key_values)
    return absorbed_draft_logits_from_target_cache(
        small_model=small_model,
        input_ids=prefix_ids,
        big_legacy_cache=big_legacy_cache,
        absorbed_state=absorbed_state,
        small_device=small_device,
        shared_layer_indices=shared_layer_indices,
        shared_variant=shared_variant,
        norm_match=norm_match,
    )


@torch.no_grad()
def greedy_target_generate(
    *,
    big_model,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    big_device: str,
) -> List[int]:
    prefix = prompt_ids.clone()
    generated: List[int] = []
    for _ in range(max_new_tokens):
        logits = target_next_logits(big_model, prefix, big_device)
        token = int(logits.argmax(dim=-1).item())
        generated.append(token)
        token_tensor = torch.tensor([[token]], dtype=prefix.dtype)
        prefix = torch.cat([prefix, token_tensor], dim=1)
    return generated


@torch.no_grad()
def greedy_speculative_decode(
    *,
    mode_name: str,
    big_model,
    small_model,
    prompt_ids: torch.Tensor,
    absorbed_state: Optional[Dict[str, Any]],
    draft_steps: int,
    max_new_tokens: int,
    big_device: str,
    small_device: str,
    topk: int,
    shared_layer_indices: Optional[Sequence[int]] = None,
    shared_variant: str = "full",
    norm_match: str = "none",
    absorbed_cache_mode: str = "recompute",
) -> Dict[str, Any]:
    if absorbed_cache_mode not in {"recompute", "prefix"}:
        raise ValueError("absorbed_cache_mode must be one of: recompute, prefix")

    prefix = prompt_ids.clone()
    generated: List[int] = []

    prompt_len = int(prompt_ids.shape[1])
    proposed_tokens = 0
    accepted_tokens = 0
    target_calls = 0
    draft_calls = 0
    full_accept_rounds = 0
    num_rounds = 0
    target_prefix_cache_reuses = 0
    round_metric_rows: List[Dict[str, float]] = []

    while len(generated) < max_new_tokens:
        num_rounds += 1
        current_prefix = prefix.clone()
        proposal: List[int] = []
        target_prefix_len = int(prefix.shape[1])
        target_prefix_legacy_cache = None

        if mode_name == "absorbed" and absorbed_cache_mode == "prefix":
            target_prefix_out = big_model(input_ids=current_prefix.to(big_device), use_cache=True)
            target_prefix_logits = target_prefix_out.logits[:, -1, :]
            target_prefix_legacy_cache = as_legacy_cache(target_prefix_out.past_key_values)
            target_calls += 1
        else:
            target_prefix_logits = target_next_logits(big_model, current_prefix, big_device)
            target_calls += 1

        for proposal_idx in range(min(draft_steps, max_new_tokens - len(generated))):
            if mode_name == "native":
                draft_logits = native_draft_next_logits(small_model, current_prefix, small_device)
            elif mode_name == "absorbed":
                if absorbed_state is None:
                    raise ValueError("absorbed_state is required when mode_name='absorbed'")
                if absorbed_cache_mode == "prefix":
                    draft_logits = absorbed_draft_logits_from_prefix_cache(
                        small_model=small_model,
                        input_ids=current_prefix,
                        target_prefix_legacy_cache=target_prefix_legacy_cache,
                        target_prefix_len=target_prefix_len,
                        absorbed_state=absorbed_state,
                        small_device=small_device,
                        shared_layer_indices=shared_layer_indices,
                        shared_variant=shared_variant,
                        norm_match=norm_match,
                    )
                    target_prefix_cache_reuses += 1
                else:
                    draft_logits = absorbed_draft_next_logits(
                        big_model=big_model,
                        small_model=small_model,
                        prefix_ids=current_prefix,
                        absorbed_state=absorbed_state,
                        big_device=big_device,
                        small_device=small_device,
                        shared_layer_indices=shared_layer_indices,
                        shared_variant=shared_variant,
                        norm_match=norm_match,
                    )
            else:
                raise ValueError(f"Unsupported mode: {mode_name}")

            draft_calls += 1
            if proposal_idx == 0:
                metrics = distribution_metrics(
                    target_prefix_logits.to(draft_logits.device),
                    draft_logits,
                    topk=topk,
                )
                round_metric_rows.append(metrics)

            token = int(draft_logits.argmax(dim=-1).item())
            proposal.append(token)
            token_tensor = torch.tensor([[token]], dtype=current_prefix.dtype)
            current_prefix = torch.cat([current_prefix, token_tensor], dim=1)

        proposed_tokens += len(proposal)
        verify_ids = current_prefix.to(big_device)
        verify_out = big_model(input_ids=verify_ids, use_cache=False)
        target_calls += 1
        verify_logits = verify_out.logits

        base_idx = int(prefix.shape[1]) - 1
        accepted_this_round = 0
        for idx, token in enumerate(proposal):
            target_token = int(verify_logits[:, base_idx + idx, :].argmax(dim=-1).item())
            if target_token != token:
                break
            accepted_this_round += 1

        accepted_tokens += accepted_this_round
        if accepted_this_round == len(proposal):
            full_accept_rounds += 1

        if accepted_this_round > 0:
            accepted_tensor = torch.tensor([proposal[:accepted_this_round]], dtype=prefix.dtype)
            prefix = torch.cat([prefix, accepted_tensor], dim=1)
            generated.extend(proposal[:accepted_this_round])

        if len(generated) >= max_new_tokens:
            break

        if accepted_this_round < len(proposal):
            correction_logits = verify_logits[:, base_idx + accepted_this_round, :]
            correction_token = int(correction_logits.argmax(dim=-1).item())
        else:
            correction_logits = verify_logits[:, -1, :]
            correction_token = int(correction_logits.argmax(dim=-1).item())

        correction_tensor = torch.tensor([[correction_token]], dtype=prefix.dtype)
        prefix = torch.cat([prefix, correction_tensor], dim=1)
        generated.append(correction_token)

    return {
        "generated_tokens": generated[:max_new_tokens],
        "prompt_len": prompt_len,
        "proposed_tokens": int(proposed_tokens),
        "accepted_tokens": int(accepted_tokens),
        "accept_rate": float(accepted_tokens / proposed_tokens) if proposed_tokens > 0 else 0.0,
        "accepted_per_round": float(accepted_tokens / num_rounds) if num_rounds > 0 else 0.0,
        "full_accept_round_fraction": float(full_accept_rounds / num_rounds) if num_rounds > 0 else 0.0,
        "target_calls": int(target_calls),
        "draft_calls": int(draft_calls),
        "num_rounds": int(num_rounds),
        "absorbed_cache_mode": absorbed_cache_mode if mode_name == "absorbed" else "native",
        "target_prefix_cache_reuses": int(target_prefix_cache_reuses),
        "target_recompute_avoided": int(target_prefix_cache_reuses if mode_name == "absorbed" and absorbed_cache_mode == "prefix" else 0),
        "round_metrics": aggregate_rows(round_metric_rows),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a greedy speculative decoding integration test comparing native draft and absorbed shared-cache draft."
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
    parser.add_argument("--draft_steps", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument(
        "--shared_layers",
        type=str,
        default="all",
        help='Which draft layers use absorbed shared-cache attention. Examples: "all", "none", "top:4", "bottom:4", "middle:4", "every_other", "0,2,4".',
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
        help="Optional per-token output normalization for absorbed full-sharing layers using calibration stats saved by fit_kv_absorbed.py.",
    )
    parser.add_argument(
        "--absorbed_cache_mode",
        type=str,
        choices=["recompute", "prefix"],
        default="recompute",
        help=(
            '"recompute" reproduces the old prototype by rebuilding target KV for every draft proposal. '
            '"prefix" computes target prefix KV once per speculative round and lets the draft keep only a temporary tail cache.'
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow_incompatible_tokenizers", action="store_true")
    parser.add_argument("--out_dir", type=str, default="outputs/absorbed_spec_eval")
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
    print(f"Absorbed cache mode: {args.absorbed_cache_mode}")

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

    per_prompt_rows: List[Dict[str, Any]] = []
    native_rows: List[Dict[str, float]] = []
    absorbed_rows: List[Dict[str, float]] = []

    for prompt_idx, block in enumerate(prompt_iter):
        prompt_ids = block.unsqueeze(0)

        target_tokens = greedy_target_generate(
            big_model=big_model,
            prompt_ids=prompt_ids,
            max_new_tokens=args.max_new_tokens,
            big_device=args.big_device,
        )

        native_result = greedy_speculative_decode(
            mode_name="native",
            big_model=big_model,
            small_model=small_model,
            prompt_ids=prompt_ids,
            absorbed_state=None,
            draft_steps=args.draft_steps,
            max_new_tokens=args.max_new_tokens,
            big_device=args.big_device,
            small_device=args.small_device,
            topk=args.topk,
            shared_layer_indices=[],
            shared_variant="full",
            norm_match="none",
            absorbed_cache_mode="recompute",
        )
        absorbed_result = greedy_speculative_decode(
            mode_name="absorbed",
            big_model=big_model,
            small_model=small_model,
            prompt_ids=prompt_ids,
            absorbed_state=absorbed_state,
            draft_steps=args.draft_steps,
            max_new_tokens=args.max_new_tokens,
            big_device=args.big_device,
            small_device=args.small_device,
            topk=args.topk,
            shared_layer_indices=shared_layer_indices,
            shared_variant=args.shared_variant,
            norm_match=args.norm_match,
            absorbed_cache_mode=args.absorbed_cache_mode,
        )

        native_match = int(native_result["generated_tokens"] == target_tokens)
        absorbed_match = int(absorbed_result["generated_tokens"] == target_tokens)

        native_row = {
            "accept_rate": native_result["accept_rate"],
            "accepted_per_round": native_result["accepted_per_round"],
            "full_accept_round_fraction": native_result["full_accept_round_fraction"],
            "matches_target_greedy": float(native_match),
            **{f"round_{k}": v for k, v in native_result["round_metrics"].items()},
        }
        absorbed_row = {
            "accept_rate": absorbed_result["accept_rate"],
            "accepted_per_round": absorbed_result["accepted_per_round"],
            "full_accept_round_fraction": absorbed_result["full_accept_round_fraction"],
            "matches_target_greedy": float(absorbed_match),
            **{f"round_{k}": v for k, v in absorbed_result["round_metrics"].items()},
        }
        native_rows.append(native_row)
        absorbed_rows.append(absorbed_row)

        per_prompt_rows.append(
            {
                "prompt_idx": int(prompt_idx),
                "native_accept_rate": native_result["accept_rate"],
                "native_accepted_tokens": native_result["accepted_tokens"],
                "native_proposed_tokens": native_result["proposed_tokens"],
                "native_matches_target_greedy": int(native_match),
                "absorbed_accept_rate": absorbed_result["accept_rate"],
                "absorbed_accepted_tokens": absorbed_result["accepted_tokens"],
                "absorbed_proposed_tokens": absorbed_result["proposed_tokens"],
                "absorbed_target_prefix_cache_reuses": absorbed_result["target_prefix_cache_reuses"],
                "absorbed_target_recompute_avoided": absorbed_result["target_recompute_avoided"],
                "absorbed_matches_target_greedy": int(absorbed_match),
            }
        )

        if wandb_run is not None:
            log_payload = {
                "eval/native_accept_rate": native_result["accept_rate"],
                "eval/absorbed_accept_rate": absorbed_result["accept_rate"],
                "eval/accept_rate_delta_absorbed_minus_native": absorbed_result["accept_rate"] - native_result["accept_rate"],
                "eval/native_accepted_per_round": native_result["accepted_per_round"],
                "eval/absorbed_accepted_per_round": absorbed_result["accepted_per_round"],
                "eval/accepted_per_round_delta_absorbed_minus_native": absorbed_result["accepted_per_round"] - native_result["accepted_per_round"],
                "eval/native_full_accept_round_fraction": native_result["full_accept_round_fraction"],
                "eval/absorbed_full_accept_round_fraction": absorbed_result["full_accept_round_fraction"],
                "eval/native_proposed_tokens": native_result["proposed_tokens"],
                "eval/absorbed_proposed_tokens": absorbed_result["proposed_tokens"],
                "eval/native_accepted_tokens": native_result["accepted_tokens"],
                "eval/absorbed_accepted_tokens": absorbed_result["accepted_tokens"],
                "eval/native_num_rounds": native_result["num_rounds"],
                "eval/absorbed_num_rounds": absorbed_result["num_rounds"],
                "eval/absorbed_target_prefix_cache_reuses": absorbed_result["target_prefix_cache_reuses"],
                "eval/absorbed_target_recompute_avoided": absorbed_result["target_recompute_avoided"],
                "eval/native_matches_target_greedy": native_match,
                "eval/absorbed_matches_target_greedy": absorbed_match,
            }
            for key, value in native_result["round_metrics"].items():
                log_payload[f"eval/native_round_{key}"] = value
            for key, value in absorbed_result["round_metrics"].items():
                log_payload[f"eval/absorbed_round_{key}"] = value
            wandb_run.log(log_payload, step=prompt_idx + 1)

        if (prompt_idx + 1) % 10 == 0:
            print(f"Processed {prompt_idx + 1} / {args.num_prompts} prompts")

    native_summary = aggregate_rows(native_rows)
    absorbed_summary = aggregate_rows(absorbed_rows)
    summary = {
        "num_prompts": len(per_prompt_rows),
        "translator_path": args.translator_path,
        "translator_output_routing_source": absorbed_state.get("output_routing_source", "unknown"),
        "shared_layers_spec": args.shared_layers,
        "shared_layer_indices": shared_layer_indices,
        "shared_variant": args.shared_variant,
        "norm_match": args.norm_match,
        "absorbed_cache_mode": args.absorbed_cache_mode,
        "native_summary": native_summary,
        "absorbed_summary": absorbed_summary,
    }

    write_csv(per_prompt_rows, os.path.join(args.out_dir, "per_prompt_rows.csv"))
    write_json(summary, os.path.join(args.out_dir, "summary.json"))

    if wandb_run is not None:
        for key, value in native_summary.items():
            wandb_run.summary[f"native/{key}"] = value
        for key, value in absorbed_summary.items():
            wandb_run.summary[f"absorbed/{key}"] = value
        wandb_run.summary["num_prompts"] = len(per_prompt_rows)
        wandb_run.summary["translator_output_routing_source"] = absorbed_state.get("output_routing_source", "unknown")
        wandb_run.summary["shared_layers_spec"] = args.shared_layers
        wandb_run.summary["num_shared_layers"] = len(shared_layer_indices)
        wandb_run.summary["shared_variant"] = args.shared_variant
        wandb_run.summary["norm_match"] = args.norm_match
        wandb_run.summary["absorbed_cache_mode"] = args.absorbed_cache_mode

    print("Done!")
    print(f"  {os.path.join(args.out_dir, 'per_prompt_rows.csv')}")
    print(f"  {os.path.join(args.out_dir, 'summary.json')}")


if __name__ == "__main__":
    main()
