#!/usr/bin/env python3
"""
fit_kv_absorbed.py

This script implements Phase 1 (Calibration & Weight Solve) of Absorptive KV-Shared 
Speculative Decoding (AKV-SD). It learns optimal Key mapping and Output mapping matrices
using an online closed-form solution (Recursive Least Squares with Woodbury matrix identity).

How to run:
    python fit_kv_absorbed.py \
        --big_model Qwen/Qwen2.5-3B \
        --small_model Qwen/Qwen2.5-1.5B \
        --dataset_name wikitext --dataset_config wikitext-2-raw-v1 \
        --train_sequences 512

Inputs:
    - Target (big) and Draft (small) models
    - Text dataset for calibration

Outputs:
    - Offline solved weights for Key transformations and Output bottle-necking.
    - Weight artifacts are saved to `out_dir`.

Logs (Weights & Biases):
    First, it trains the Key transformations per layer over streamed batches.
    It logs (for each step/batch):
        - K_L2_distance: L2 distance between original and reconstructed Keys.
        - K_max_diff: Magnitude of the maximum coordinate difference.
        - K_cosine_similarity: Cosine similarity of original and reconstructed Keys.

    Next, it trains the Attention Output transformations per layer.
    It logs (for each step/batch):
        - O_L2_distance: L2 distance between original and reconstructed Attention Output.
        - O_max_diff: Magnitude of the maximum coordinate difference.
        - O_cosine_similarity: Cosine similarity of original and reconstructed Attention Output.
"""

import argparse
import atexit
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from kv_utils import (
    as_legacy_cache,
    depth_layer_map,
    flatten_kv,
    get_head_dim,
    get_kv_dim,
    get_num_kv_heads,
    iter_token_blocks,
    load_causal_lm,
    load_tokenizer,
    set_seed,
    tokenizer_compatibility_report,
    unflatten_kv,
    write_json,
)

def init_wandb(args: argparse.Namespace) -> Optional[any]:
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
        wandb.finish()
    atexit.register(_finish_wandb)
    return run

class OnlineRidgeAccumulator:
    """Uses the Woodbury matrix identity to compute Recursive Least Squares (RLS)."""
    def __init__(self, d_in: int, d_out: int, lambda_reg: float = 1e-4, device: str = "cpu"):
        self.d_in = d_in
        self.d_out = d_out
        self.device = device
        # P = (X^T X + \lambda I)^{-1}
        self.P = (torch.eye(d_in + 1, dtype=torch.float64, device=device) / lambda_reg)
        self.P[-1, -1] = 1.0  # bias regularization 
        self.W = torch.zeros(d_in + 1, d_out, dtype=torch.float64, device=device)
        self.num_rows = 0

    def update(self, x: torch.Tensor, y: torch.Tensor):
        x = x.to(dtype=torch.float64, device=self.device)
        y = y.to(dtype=torch.float64, device=self.device)
        N = x.shape[0]
        x_aug = torch.cat([x, torch.ones(N, 1, dtype=torch.float64, device=self.device)], dim=1)
        
        d = self.d_in + 1
        if N < d:
            # Woodbury matrix identity (faster when batch size N < parameters d)
            I_N = torch.eye(N, dtype=torch.float64, device=self.device)
            S = I_N + x_aug @ self.P @ x_aug.T
            S_inv = torch.linalg.solve(S, I_N)
            K = self.P @ x_aug.T @ S_inv
            
            self.P = self.P - K @ x_aug @ self.P
            err = y - x_aug @ self.W
            self.W = self.W + K @ err
        else:
            # Sherman-Morrison rank-1 updates (faster when batch size N >= parameters d)
            for i in range(N):
                xi = x_aug[i:i+1]  # [1, d]
                yi = y[i:i+1]      # [1, d_out]
                
                denom = 1.0 + (xi @ self.P @ xi.T)
                K_i = (self.P @ xi.T) / denom
                
                self.P = self.P - K_i @ (xi @ self.P)
                err_i = yi - xi @ self.W
                self.W = self.W + K_i @ err_i
                
        self.num_rows += N

    def get_weights(self) -> Tuple[torch.Tensor, torch.Tensor]:
        weight = self.W[:-1].T.to(dtype=torch.float32).contiguous() # [d_out, d_in]
        bias = self.W[-1].to(dtype=torch.float32).contiguous()      # [d_out]
        return weight, bias

def compute_stats(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    pred = pred.float()
    target = target.float()
    l2_dist = float(torch.norm(pred - target, p=2, dim=-1).mean().item())
    max_diff = float(torch.abs(pred - target).max().item())
    cos_sim = float(F.cosine_similarity(pred, target, dim=-1).mean().item())
    return {"L2_distance": l2_dist, "max_diff": max_diff, "cosine_similarity": cos_sim}


def load_layer_map(path: str, *, num_big_layers: int, num_small_layers: int) -> List[int]:
    """Load a learned draft-layer -> target-layer map from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    raw_map = payload.get("layer_map", payload) if isinstance(payload, dict) else payload
    if not isinstance(raw_map, list):
        raise ValueError("--layer_map_file must contain a JSON list or an object with a 'layer_map' list.")
    layer_map = [int(x) for x in raw_map]
    if len(layer_map) != num_small_layers:
        raise ValueError(
            f"Layer map length {len(layer_map)} does not match draft layer count {num_small_layers}."
        )
    for idx, target_idx in enumerate(layer_map):
        if target_idx < 0 or target_idx >= num_big_layers:
            raise ValueError(
                f"Layer map entry {idx}->{target_idx} is out of range for target layers={num_big_layers}."
            )
    if any(layer_map[i] > layer_map[i + 1] for i in range(len(layer_map) - 1)):
        raise ValueError("Layer map must be monotonic nondecreasing.")
    return layer_map


def scalar_tensor_stats(tensor: torch.Tensor) -> Dict[str, float]:
    tensor = tensor.float()
    mean = float(tensor.mean().item())
    std = float(tensor.std(unbiased=False).item())
    rms = float(torch.sqrt(torch.mean(tensor.square())).item())
    abs_mean = float(tensor.abs().mean().item())
    return {"mean": mean, "std": std, "rms": rms, "abs_mean": abs_mean}


class RunningTensorStats:
    """Accumulates scalar activation stats without storing calibration tensors."""

    def __init__(self) -> None:
        self.count = 0
        self.sum = 0.0
        self.sum_sq = 0.0
        self.abs_sum = 0.0

    def update(self, tensor: torch.Tensor) -> None:
        tensor = tensor.detach().float()
        self.count += int(tensor.numel())
        self.sum += float(tensor.sum().item())
        self.sum_sq += float(tensor.square().sum().item())
        self.abs_sum += float(tensor.abs().sum().item())

    def as_dict(self) -> Dict[str, float]:
        if self.count == 0:
            return {"count": 0, "mean": 0.0, "std": 0.0, "rms": 0.0, "abs_mean": 0.0}
        mean = self.sum / self.count
        mean_sq = self.sum_sq / self.count
        var = max(0.0, mean_sq - mean * mean)
        return {
            "count": int(self.count),
            "mean": float(mean),
            "std": float(var ** 0.5),
            "rms": float(mean_sq ** 0.5),
            "abs_mean": float(self.abs_sum / self.count),
        }


def affine_apply(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    # x: [..., d_in], weight: [d_out, d_in], bias: [d_out]
    return F.linear(x.float(), weight.float(), bias.float())


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_q_only(
    q: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> torch.Tensor:
    cos = cos.unsqueeze(unsqueeze_dim).to(dtype=q.dtype, device=q.device)
    sin = sin.unsqueeze(unsqueeze_dim).to(dtype=q.dtype, device=q.device)
    return (q * cos) + (rotate_half(q) * sin)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_key_value_heads, seqlen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, seqlen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, seqlen, head_dim)


def compute_shared_attention_weights(
    *,
    attn_module,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    mapped_key_states: torch.Tensor,
) -> torch.Tensor:
    # In newer transformers, Qwen2Attention no longer exposes num_heads / head_dim
    # directly on the module; they live on the config. Fall back gracefully.
    config = getattr(attn_module, "config", None)
    num_heads = getattr(attn_module, "num_heads", None)
    if num_heads is None and config is not None:
        num_heads = config.num_attention_heads
    head_dim = getattr(attn_module, "head_dim", None)
    if head_dim is None and config is not None:
        head_dim = getattr(config, "head_dim", None) or (config.hidden_size // num_heads)

    bsz, seq_len, _ = hidden_states.shape
    hidden_shape = (bsz, seq_len, num_heads, head_dim)
    query_states = attn_module.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states = apply_rotary_pos_emb_q_only(query_states, cos, sin)

    num_kv_heads = mapped_key_states.shape[1]
    if num_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_heads={num_heads} is not divisible by mapped num_kv_heads={num_kv_heads}"
        )
    key_states = repeat_kv(mapped_key_states, num_heads // num_kv_heads)
    scaling = float(getattr(attn_module, "scaling", head_dim ** -0.5))

    attn_scores = torch.matmul(query_states.float(), key_states.transpose(2, 3).float()) * scaling
    if attention_mask is not None:
        attn_scores = attn_scores + attention_mask.to(device=attn_scores.device, dtype=attn_scores.dtype)
    attn_weights = torch.softmax(attn_scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
    return attn_weights

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--train_sequences", type=int, default=512)
    parser.add_argument("--lambda_reg", type=float, default=1e-4)
    parser.add_argument(
        "--output_routing_source",
        type=str,
        choices=["native", "shared"],
        default="shared",
        help="Use native draft attentions or recompute shared attentions from the learned key map when solving O.",
    )
    parser.add_argument(
        "--layer_map_file",
        type=str,
        default=None,
        help="Optional JSON file from learn_layer_map.py. Falls back to depth-based mapping when omitted.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shuffle_train", action="store_true")
    parser.add_argument("--stream_train", action="store_true")
    parser.add_argument("--allow_incompatible_tokenizers", action="store_true")
    parser.add_argument("--out_dir", type=str, default="outputs/kv_absorbed")
    
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
    if (not compatibility["all_probe_encodings_match"]) and (not args.allow_incompatible_tokenizers):
        raise ValueError("Tokenizers appear incompatible.")

    big_model = load_causal_lm(args.big_model, device=args.big_device, dtype_name=args.big_dtype, attn_implementation="eager")
    small_model = load_causal_lm(args.small_model, device=args.small_device, dtype_name=args.small_dtype, attn_implementation="eager")

    num_big_layers = int(big_model.config.num_hidden_layers)
    num_small_layers = int(small_model.config.num_hidden_layers)
    big_kv_dim = get_kv_dim(big_model.config)
    small_kv_dim = get_kv_dim(small_model.config)
    small_kv_heads = get_num_kv_heads(small_model.config)
    small_q_heads = small_model.config.num_attention_heads
    small_head_dim = get_head_dim(small_model.config)
    small_d_model = small_model.config.hidden_size

    if args.layer_map_file is not None:
        layer_map = load_layer_map(args.layer_map_file, num_big_layers=num_big_layers, num_small_layers=num_small_layers)
        layer_map_source = args.layer_map_file
    else:
        layer_map = depth_layer_map(num_big_layers, num_small_layers)
        layer_map_source = "depth"
    print(f"Layer map: {layer_map}")
    print(f"Layer map source: {layer_map_source}")
    print(f"Output routing source: {args.output_routing_source}")

    # Intercept Y_draft (output of self_attn block)
    draft_attention_outputs = {}
    draft_attention_inputs: Dict[int, Dict[str, Any]] = {}
    def get_attn_hook(layer_idx):
        def hook(module, args, kwargs, output):
            # For causal LM, self_attn output is a tuple (attn_output, attn_weights, past_key_value)
            draft_attention_outputs[layer_idx] = output[0].detach()
            hidden_states = kwargs.get("hidden_states")
            if hidden_states is None and len(args) > 0:
                hidden_states = args[0]
            attention_mask = kwargs.get("attention_mask")
            position_embeddings = kwargs.get("position_embeddings")
            draft_attention_inputs[layer_idx] = {
                "hidden_states": hidden_states.detach() if hidden_states is not None else None,
                "attention_mask": attention_mask.detach() if attention_mask is not None else None,
                "position_embeddings": tuple(t.detach() for t in position_embeddings) if position_embeddings is not None else None,
            }
        return hook

    # Register hooks on small model's attention layers
    hooks = []
    for i, layer in enumerate(small_model.model.layers):
        h = layer.self_attn.register_forward_hook(get_attn_hook(i), with_kwargs=True)
        hooks.append(h)

    # ---------------------------------------------------------------------------------
    # Phase 1.1: Train Key Transformations
    # ---------------------------------------------------------------------------------
    print("\n[Part 1] Training Key Transformations...")
    k_accs = [OnlineRidgeAccumulator(big_kv_dim, small_kv_dim, lambda_reg=args.lambda_reg, device=args.small_device) 
              for _ in range(num_small_layers)]
    
    train_iter = iter_token_blocks(
        tokenizer=big_tokenizer, seq_len=args.seq_len, max_blocks=args.train_sequences,
        dataset_name=args.dataset_name, dataset_config=args.dataset_config,
        split=args.train_split, text_file=args.text_file, text_column=args.text_column,
        shuffle=args.shuffle_train, seed=args.seed, streaming=args.stream_train,
    )

    for step, block in enumerate(train_iter):
        full_ids_big = block.unsqueeze(0).to(args.big_device)
        full_ids_small = block.unsqueeze(0).to(args.small_device)
        
        with torch.no_grad():
            big_out = big_model(input_ids=full_ids_big, use_cache=True)
            small_out = small_model(input_ids=full_ids_small, use_cache=True)
            
        big_legacy = as_legacy_cache(big_out.past_key_values)
        small_legacy = as_legacy_cache(small_out.past_key_values)

        log_dict = {}
        for small_layer_idx, big_layer_idx in enumerate(layer_map):
            k_big, v_big = big_legacy[big_layer_idx]
            k_small, v_small = small_legacy[small_layer_idx]
            
            xk = flatten_kv(k_big).float().to(args.small_device)
            yk = flatten_kv(k_small).float().to(args.small_device)
            
            # Predict with current weights
            weight, bias = k_accs[small_layer_idx].get_weights()
            pred_k = affine_apply(xk, weight, bias)
            
            # Stats
            stats = compute_stats(pred_k, yk)
            for k, v in stats.items():
                log_dict[f"K_{k}/layer_{small_layer_idx}"] = v
            
            # Update accumulator
            k_accs[small_layer_idx].update(xk, yk)
            
        if wandb_run is not None:
            wandb_run.log(log_dict, step=step+1)
        if (step+1) % 20 == 0:
            print(f"  Key Phase: sequence {step + 1} / {args.train_sequences}")

    # ---------------------------------------------------------------------------------
    # Phase 1.2: Train Attention Output Transformations
    # ---------------------------------------------------------------------------------
    print("\n[Part 2] Training Attention Layer Output Transformations...")
    o_accs = [OnlineRidgeAccumulator(small_q_heads * small_head_dim, small_d_model, lambda_reg=args.lambda_reg, device=args.small_device) 
              for _ in range(num_small_layers)]
    o_target_stats = [RunningTensorStats() for _ in range(num_small_layers)]
    o_pred_stats = [RunningTensorStats() for _ in range(num_small_layers)]
    o_input_stats = [RunningTensorStats() for _ in range(num_small_layers)]
    
    # Reset iterator
    train_iter = iter_token_blocks(
        tokenizer=big_tokenizer, seq_len=args.seq_len, max_blocks=args.train_sequences,
        dataset_name=args.dataset_name, dataset_config=args.dataset_config,
        split=args.train_split, text_file=args.text_file, text_column=args.text_column,
        shuffle=args.shuffle_train, seed=args.seed, streaming=args.stream_train,
    )

    for step, block in enumerate(train_iter):
        full_ids_big = block.unsqueeze(0).to(args.big_device)
        full_ids_small = block.unsqueeze(0).to(args.small_device)
        
        with torch.no_grad():
            big_out = big_model(input_ids=full_ids_big, use_cache=True)
            small_out = small_model(input_ids=full_ids_small, use_cache=True, output_attentions=True)
            
        big_legacy = as_legacy_cache(big_out.past_key_values)
        attentions = small_out.attentions

        log_dict = {}
        for small_layer_idx, big_layer_idx in enumerate(layer_map):
            k_big, v_big = big_legacy[big_layer_idx]
            k_big = k_big.to(args.small_device)
            v_big = v_big.to(args.small_device) # [B, target_kv_heads, Seq, head_dim]
            
            A_draft = attentions[small_layer_idx].to(args.small_device) # [B, draft_q_heads, Seq, Seq]
            Y_draft = draft_attention_outputs[small_layer_idx].to(args.small_device) # [B, Seq, d_model]
            layer_inputs = draft_attention_inputs.get(small_layer_idx, {})
            hidden_states_in = layer_inputs.get("hidden_states")
            attention_mask = layer_inputs.get("attention_mask")
            position_embeddings = layer_inputs.get("position_embeddings")
            if hidden_states_in is None or position_embeddings is None:
                raise RuntimeError("Missing draft attention inputs needed to compute the selected output routing source.")
            
            bsz, target_kv_heads, seq_len, head_dim = v_big.shape
            draft_q_heads = A_draft.shape[1] 
            
            if draft_q_heads % target_kv_heads != 0:
                raise ValueError("Draft Q heads must be divisible by Target KV heads for GQA alignment.")
            repeats = draft_q_heads // target_kv_heads

            if args.output_routing_source == "shared":
                k_weight, k_bias = k_accs[small_layer_idx].get_weights()
                k_shared_flat = affine_apply(flatten_kv(k_big), k_weight, k_bias)
                k_shared = unflatten_kv(k_shared_flat, bsz, seq_len, small_kv_heads, small_head_dim)
                A_routing = compute_shared_attention_weights(
                    attn_module=small_model.model.layers[small_layer_idx].self_attn,
                    hidden_states=hidden_states_in.to(args.small_device),
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    mapped_key_states=k_shared.to(args.small_device),
                )
            else:
                A_routing = A_draft
            
            # Expand V_target to match drafted q_heads
            v_expanded = v_big.unsqueeze(2).expand(bsz, target_kv_heads, repeats, seq_len, head_dim)
            v_expanded = v_expanded.reshape(bsz, draft_q_heads, seq_len, head_dim)
            
            # H_target_tilde = A_routing @ V_expanded
            H_target_tilde = torch.matmul(A_routing.float(), v_expanded.float()) # [B, draft_q_heads, Seq, head_dim]
            
            # Flatten to [B*Seq, draft_q_heads * head_dim]
            # Must transpose shape from [B, H, Seq, D] to [B, Seq, H, D] then flatten
            H_target_tilde = H_target_tilde.transpose(1, 2).reshape(bsz * seq_len, draft_q_heads * head_dim)
            Y_draft_flat = Y_draft.float().reshape(bsz * seq_len, small_d_model)
            
            # Predict with current weights
            weight, bias = o_accs[small_layer_idx].get_weights()
            pred_Y = affine_apply(H_target_tilde, weight, bias)
            
            # Stats
            stats = compute_stats(pred_Y, Y_draft_flat)
            for k, v in stats.items():
                log_dict[f"O_{k}/layer_{small_layer_idx}"] = v
            target_scale = scalar_tensor_stats(Y_draft_flat)
            pred_scale = scalar_tensor_stats(pred_Y)
            input_scale = scalar_tensor_stats(H_target_tilde)
            o_target_stats[small_layer_idx].update(Y_draft_flat)
            o_pred_stats[small_layer_idx].update(pred_Y)
            o_input_stats[small_layer_idx].update(H_target_tilde)
            for k, v in target_scale.items():
                log_dict[f"O_target_{k}/layer_{small_layer_idx}"] = v
            for k, v in pred_scale.items():
                log_dict[f"O_pred_{k}/layer_{small_layer_idx}"] = v
            for k, v in input_scale.items():
                log_dict[f"O_input_{k}/layer_{small_layer_idx}"] = v
            log_dict[f"O_pred_to_target_rms_ratio/layer_{small_layer_idx}"] = (
                pred_scale["rms"] / max(target_scale["rms"], 1e-12)
            )
            log_dict[f"O_pred_to_target_std_ratio/layer_{small_layer_idx}"] = (
                pred_scale["std"] / max(target_scale["std"], 1e-12)
            )
            
            # Update accumulator
            o_accs[small_layer_idx].update(H_target_tilde, Y_draft_flat)
            
        # The offset here makes the step index continue after K training steps for cleanly separated charts
        if wandb_run is not None:
            wandb_run.log(log_dict, step=args.train_sequences + step + 1)
            
        if (step+1) % 20 == 0:
            print(f"  Output Phase: sequence {step + 1} / {args.train_sequences}")

    print("\nSaving solved weights...")
    k_weights, k_biases = [], []
    o_weights, o_biases = [], []
    for acc in k_accs:
        w, b = acc.get_weights()
        k_weights.append(w.cpu())
        k_biases.append(b.cpu())
    for acc in o_accs:
        w, b = acc.get_weights()
        o_weights.append(w.cpu())
        o_biases.append(b.cpu())

    o_target_stats_dicts = [stats.as_dict() for stats in o_target_stats]
    o_pred_stats_dicts = [stats.as_dict() for stats in o_pred_stats]
    o_input_stats_dicts = [stats.as_dict() for stats in o_input_stats]
    o_target_mean = [stats["mean"] for stats in o_target_stats_dicts]
    o_target_std = [stats["std"] for stats in o_target_stats_dicts]
    o_target_rms = [stats["rms"] for stats in o_target_stats_dicts]

    state = {
        "big_model": args.big_model, "small_model": args.small_model, "layer_map": layer_map,
        "k_weights": k_weights, "k_biases": k_biases,
        "o_weights": o_weights, "o_biases": o_biases,
        "o_target_mean": o_target_mean,
        "o_target_std": o_target_std,
        "o_target_rms": o_target_rms,
        "o_target_stats": o_target_stats_dicts,
        "o_pred_stats": o_pred_stats_dicts,
        "o_input_stats": o_input_stats_dicts,
        "lambda_reg": float(args.lambda_reg),
        "output_routing_source": args.output_routing_source,
        "layer_map_source": layer_map_source,
        "small_q_heads": int(small_q_heads),
        "small_kv_heads": int(small_kv_heads),
        "small_head_dim": int(small_head_dim),
        "small_d_model": int(small_d_model),
    }
    torch.save(state, os.path.join(args.out_dir, "absorbed_translator.pt"))
    write_json(
        {
            "big_model": args.big_model,
            "small_model": args.small_model,
            "layer_map": layer_map,
            "layer_map_source": layer_map_source,
            "lambda_reg": float(args.lambda_reg),
            "output_routing_source": args.output_routing_source,
            "small_q_heads": int(small_q_heads),
            "small_kv_heads": int(small_kv_heads),
            "small_head_dim": int(small_head_dim),
            "small_d_model": int(small_d_model),
            "train_sequences": int(args.train_sequences),
            "o_target_stats": o_target_stats_dicts,
            "o_pred_stats": o_pred_stats_dicts,
            "o_input_stats": o_input_stats_dicts,
        },
        os.path.join(args.out_dir, "summary.json"),
    )
    if wandb_run is not None:
        for layer_idx, stats in enumerate(o_target_stats_dicts):
            for key, value in stats.items():
                wandb_run.summary[f"O_target_calibration/{key}/layer_{layer_idx}"] = value
        for layer_idx, stats in enumerate(o_pred_stats_dicts):
            for key, value in stats.items():
                wandb_run.summary[f"O_pred_calibration/{key}/layer_{layer_idx}"] = value
        for layer_idx, stats in enumerate(o_input_stats_dicts):
            for key, value in stats.items():
                wandb_run.summary[f"O_input_calibration/{key}/layer_{layer_idx}"] = value
    print("Done!")
    
    # Remove hooks
    for h in hooks:
        h.remove()

if __name__ == "__main__":
    main()
