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
import os
import atexit
from typing import Dict, List, Optional, Tuple

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

def affine_apply(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    # x: [..., d_in], weight: [d_out, d_in], bias: [d_out]
    return F.linear(x.float(), weight.float(), bias.float())

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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shuffle_train", action="store_true")
    parser.add_argument("--allow_incompatible_tokenizers", action="store_true")
    parser.add_argument("--out_dir", type=str, default="outputs/kv_absorbed")
    
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
    small_q_heads = small_model.config.num_attention_heads
    small_head_dim = get_head_dim(small_model.config)
    small_d_model = small_model.config.hidden_size

    layer_map = depth_layer_map(num_big_layers, num_small_layers)
    print(f"Layer map: {layer_map}")

    # Intercept Y_draft (output of self_attn block)
    draft_attention_outputs = {}
    def get_attn_hook(layer_idx):
        def hook(module, input, output):
            # For causal LM, self_attn output is a tuple (attn_output, attn_weights, past_key_value)
            draft_attention_outputs[layer_idx] = output[0].detach()
        return hook

    # Register hooks on small model's attention layers
    hooks = []
    for i, layer in enumerate(small_model.model.layers):
        h = layer.self_attn.register_forward_hook(get_attn_hook(i))
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
        shuffle=args.shuffle_train, seed=args.seed,
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
    
    # Reset iterator
    train_iter = iter_token_blocks(
        tokenizer=big_tokenizer, seq_len=args.seq_len, max_blocks=args.train_sequences,
        dataset_name=args.dataset_name, dataset_config=args.dataset_config,
        split=args.train_split, text_file=args.text_file, text_column=args.text_column,
        shuffle=args.shuffle_train, seed=args.seed,
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
            _, v_big = big_legacy[big_layer_idx]
            v_big = v_big.to(args.small_device) # [B, target_kv_heads, Seq, head_dim]
            
            A_draft = attentions[small_layer_idx].to(args.small_device) # [B, draft_q_heads, Seq, Seq]
            Y_draft = draft_attention_outputs[small_layer_idx].to(args.small_device) # [B, Seq, d_model]
            
            bsz, target_kv_heads, seq_len, head_dim = v_big.shape
            draft_q_heads = A_draft.shape[1] 
            
            if draft_q_heads % target_kv_heads != 0:
                raise ValueError("Draft Q heads must be divisible by Target KV heads for GQA alignment.")
            repeats = draft_q_heads // target_kv_heads
            
            # Expand V_target to match drafted q_heads
            v_expanded = v_big.unsqueeze(2).expand(bsz, target_kv_heads, repeats, seq_len, head_dim)
            v_expanded = v_expanded.reshape(bsz, draft_q_heads, seq_len, head_dim)
            
            # H_target_tilde = A_draft @ V_expanded
            H_target_tilde = torch.matmul(A_draft.float(), v_expanded.float()) # [B, draft_q_heads, Seq, head_dim]
            
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

    state = {
        "big_model": args.big_model, "small_model": args.small_model, "layer_map": layer_map,
        "k_weights": k_weights, "k_biases": k_biases,
        "o_weights": o_weights, "o_biases": o_biases,
    }
    torch.save(state, os.path.join(args.out_dir, "absorbed_translator.pt"))
    print("Done!")
    
    # Remove hooks
    for h in hooks:
        h.remove()

if __name__ == "__main__":
    main()
