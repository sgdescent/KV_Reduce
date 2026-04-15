#!/usr/bin/env python3
import argparse
import atexit
import csv
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from kv_utils import (
    RidgeAccumulator,
    affine_apply,
    as_legacy_cache,
    depth_layer_map,
    distribution_metrics,
    first_mismatch_position,
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


class ResidualMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.ln(x))


def solve_accumulators(accumulators: List[RidgeAccumulator]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    weights, biases = [], []
    for acc in accumulators:
        w, b = acc.solve()
        weights.append(w)
        biases.append(b)
    return weights, biases


def translate_ridge_legacy_cache(
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


def translate_hybrid_legacy_cache(
    big_legacy_cache,
    layer_map: List[int],
    k_weights: List[torch.Tensor],
    k_biases: List[torch.Tensor],
    v_mlps: List[nn.Module],
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
        yv = v_mlps[small_layer_idx](xv)

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


def metric_prefix_dict(prefix: str, ref_logits: torch.Tensor, test_logits: torch.Tensor, topk: int) -> Dict[str, float]:
    # Fast eval path: only compute top-1 token agreement.
    return {
        f"{prefix}_top1_match": float(
            (test_logits.argmax(dim=-1) == ref_logits.argmax(dim=-1)).float().mean().item()
        )
    }



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit ridge K + MLP V KV-cache translator from a big model to a small model.")
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
    parser.add_argument("--generate_steps", type=int, default=16, help="Ignored in this fast-eval version; kept only for CLI compatibility.")

    parser.add_argument("--v_rows_per_layer", type=int, default=25000)
    parser.add_argument("--v_hidden_dim", type=int, default=1024)
    parser.add_argument("--v_dropout", type=float, default=0.0)
    parser.add_argument("--v_epochs", type=int, default=5)
    parser.add_argument("--v_batch_size", type=int, default=1024)
    parser.add_argument("--v_lr", type=float, default=1e-3)
    parser.add_argument("--v_weight_decay", type=float, default=1e-4)
    parser.add_argument("--v_cos_loss_weight", type=float, default=0.1)
    parser.add_argument("--v_val_frac", type=float, default=0.05)

    parser.add_argument("--out_dir", type=str, default="outputs/kv_hybrid_vmlp_probe")

    parser.add_argument("--wandb", action="store_true", help="Log metrics to Weights & Biases (set WANDB_API_KEY in the environment).")
    parser.add_argument("--wandb_project", type=str, default="kv-reduce", help="W&B project name.")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name (default: auto).")
    parser.add_argument("--wandb_entity", type=str, default=None, help="W&B entity (team or user). Optional.")
    parser.add_argument("--wandb_group", type=str, default=None, help="Optional W&B group for sweeps / grouping runs.")
    return parser


def init_wandb(args: argparse.Namespace) -> Optional[Any]:
    if not args.wandb:
        return None
    try:
        import wandb
    except ImportError as e:
        raise ImportError("W&B logging requested but wandb is not installed. Install with: pip install wandb") from e

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


def train_v_mlps(
    v_x_by_layer: List[torch.Tensor],
    v_y_by_layer: List[torch.Tensor],
    device: str,
    dim: int,
    hidden_dim: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    cos_loss_weight: float,
    val_frac: float,
    wandb_run: Optional[Any] = None,
    wandb_log_step_offset: int = 0,
) -> Tuple[List[nn.Module], List[Dict[str, float]]]:
    models: List[nn.Module] = []
    summaries: List[Dict[str, float]] = []
    num_layers = len(v_x_by_layer)

    layer_bar = tqdm(
        enumerate(zip(v_x_by_layer, v_y_by_layer)),
        total=num_layers,
        desc="V-MLP layers",
        unit="layer",
    )
    for layer_idx, (x_cpu, y_cpu) in layer_bar:
        if x_cpu.numel() == 0:
            raise ValueError(f"No V training rows collected for layer {layer_idx}. Increase --v_rows_per_layer or train data.")

        n = x_cpu.shape[0]
        n_val = 0 if n < 20 else max(1, int(val_frac * n))
        perm = torch.randperm(n)
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]

        x_train = x_cpu[train_idx]
        y_train = y_cpu[train_idx]
        x_val = x_cpu[val_idx] if n_val > 0 else None
        y_val = y_cpu[val_idx] if n_val > 0 else None

        model = ResidualMLP(dim=dim, hidden_dim=hidden_dim, dropout=dropout).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        best_state = None
        best_val = float("inf")
        best_at_epoch = -1
        last_train = float("nan")
        first_train = float("nan")
        first_val = float("nan")
        final_val = float("nan")

        epoch_bar = tqdm(
            range(epochs),
            desc=f"  layer {layer_idx + 1}/{num_layers}",
            leave=False,
            unit="epoch",
        )
        for epoch in epoch_bar:
            model.train()
            order = torch.randperm(x_train.shape[0])
            total_train_loss = 0.0
            total_train_rows = 0

            for start in range(0, x_train.shape[0], batch_size):
                idx = order[start : start + batch_size]
                xb = x_train[idx].to(device=device, dtype=torch.float32, non_blocking=True)
                yb = y_train[idx].to(device=device, dtype=torch.float32, non_blocking=True)

                pred = model(xb)
                mse = F.mse_loss(pred, yb)
                cos = 1.0 - F.cosine_similarity(pred, yb, dim=-1).mean()
                loss = mse + cos_loss_weight * cos

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                total_train_loss += float(loss.item()) * xb.shape[0]
                total_train_rows += int(xb.shape[0])

            last_train = total_train_loss / max(1, total_train_rows)
            if epoch == 0:
                first_train = last_train

            model.eval()
            if n_val > 0:
                total_val_loss = 0.0
                total_val_rows = 0
                with torch.no_grad():
                    for start in range(0, x_val.shape[0], batch_size):
                        xb = x_val[start : start + batch_size].to(device=device, dtype=torch.float32, non_blocking=True)
                        yb = y_val[start : start + batch_size].to(device=device, dtype=torch.float32, non_blocking=True)
                        pred = model(xb)
                        mse = F.mse_loss(pred, yb)
                        cos = 1.0 - F.cosine_similarity(pred, yb, dim=-1).mean()
                        loss = mse + cos_loss_weight * cos
                        total_val_loss += float(loss.item()) * xb.shape[0]
                        total_val_rows += int(xb.shape[0])
                val_loss = total_val_loss / max(1, total_val_rows)
            else:
                val_loss = last_train

            if epoch == 0:
                first_val = val_loss
            final_val = val_loss

            improved = val_loss < best_val
            if improved:
                best_val = val_loss
                best_at_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            epoch_bar.set_postfix(
                train=f"{last_train:.5f}",
                val=f"{val_loss:.5f}",
                best=f"{best_val:.5f}",
                ep_best=best_at_epoch,
                ok="*" if improved else "",
                refresh=True,
            )

            if wandb_run is not None:
                # Offset so V-MLP curves appear after data-collection steps on the same W&B x-axis.
                step = wandb_log_step_offset + layer_idx * epochs + epoch
                wandb_run.log(
                    {
                        f"v_mlp/layer_{layer_idx}/train_loss": last_train,
                        f"v_mlp/layer_{layer_idx}/val_loss": val_loss,
                        f"v_mlp/layer_{layer_idx}/best_val_loss": best_val,
                    },
                    step=step,
                )

        epoch_bar.close()

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        models.append(model)

        train_delta = float(first_train - last_train) if first_train == first_train else float("nan")
        val_delta = float(first_val - final_val) if (n_val > 0 and first_val == first_val) else float("nan")
        layer_bar.set_postfix(
            best=f"{best_val:.5f}",
            d_trn=f"{train_delta:.5f}",
            d_val=(f"{val_delta:.5f}" if n_val > 0 else "n/a"),
        )

        summaries.append(
            {
                "layer_idx": layer_idx,
                "num_rows": int(n),
                "num_train_rows": int(x_train.shape[0]),
                "num_val_rows": int(n_val),
                "epochs_ran": int(epochs),
                "first_train_loss": float(first_train),
                "last_train_loss": float(last_train),
                "train_loss_drop": float(train_delta),
                "first_val_loss": float(first_val) if n_val > 0 else float("nan"),
                "last_val_loss": float(final_val) if n_val > 0 else float("nan"),
                "val_loss_drop": float(val_delta) if n_val > 0 else float("nan"),
                "best_val_loss": float(best_val),
                "best_at_epoch": int(best_at_epoch),
            }
        )

    layer_bar.close()
    return models, summaries


def main() -> None:
    args = build_parser().parse_args()
    if args.seq_len < 2:
        raise ValueError("seq_len must be at least 2")
    if args.position_stride < 1:
        raise ValueError("position_stride must be >= 1")
    if args.v_rows_per_layer < 1:
        raise ValueError("v_rows_per_layer must be >= 1")

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    wandb_run = init_wandb(args)

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

    v_x_parts: List[List[torch.Tensor]] = [[] for _ in range(num_small_layers)]
    v_y_parts: List[List[torch.Tensor]] = [[] for _ in range(num_small_layers)]
    v_counts = [0 for _ in range(num_small_layers)]

    print("\nCollecting train statistics for ridge K/ridge V + sampled rows for MLP V...")
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

    train_idx_last = -1
    for train_idx, block in enumerate(train_iter):
        train_idx_last = train_idx
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

            remaining = args.v_rows_per_layer - v_counts[small_layer_idx]
            if remaining > 0:
                take = min(remaining, xv.shape[0])
                # xv from big cache (big_device) vs yv from small cache (small_device): index on CPU.
                xv_cpu = xv.detach().cpu()
                yv_cpu = yv.detach().cpu()
                if take < xv.shape[0]:
                    perm = torch.randperm(xv.shape[0])[:take]
                    xv_keep = xv_cpu[perm]
                    yv_keep = yv_cpu[perm]
                else:
                    xv_keep = xv_cpu
                    yv_keep = yv_cpu
                v_x_parts[small_layer_idx].append(xv_keep.to(torch.float16))
                v_y_parts[small_layer_idx].append(yv_keep.to(torch.float16))
                v_counts[small_layer_idx] += int(take)

        if train_idx % 20 == 0:
            filled = sum(int(c >= args.v_rows_per_layer) for c in v_counts)
            print(f"  train sequence {train_idx + 1} / {args.train_sequences} | sampled V layers filled: {filled}/{num_small_layers}")
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "collection/sequence_idx": train_idx + 1,
                        "collection/v_layers_filled": filled,
                        "collection/v_layer_counts_min": min(v_counts) if v_counts else 0,
                    },
                    step=train_idx,
                )

    num_train_sequences = train_idx_last + 1 if train_idx_last >= 0 else 0
    print(f"  Data collection done: {num_train_sequences} sequence(s) of length {args.seq_len} (full dataset pass or --train_sequences cap).")

    print("\nSolving ridge regressions...")
    k_weights, k_biases = solve_accumulators(k_accs)
    v_weights, v_biases = solve_accumulators(v_accs)

    v_x_by_layer = [torch.cat(parts, dim=0).to(torch.float32) if parts else torch.empty(0, big_kv_dim, dtype=torch.float32) for parts in v_x_parts]
    v_y_by_layer = [torch.cat(parts, dim=0).to(torch.float32) if parts else torch.empty(0, small_kv_dim, dtype=torch.float32) for parts in v_y_parts]

    print("\nTraining per-layer MLPs for V...")
    v_mlps, v_mlp_summaries = train_v_mlps(
        v_x_by_layer=v_x_by_layer,
        v_y_by_layer=v_y_by_layer,
        device=args.small_device,
        dim=small_kv_dim,
        hidden_dim=args.v_hidden_dim,
        dropout=args.v_dropout,
        epochs=args.v_epochs,
        batch_size=args.v_batch_size,
        lr=args.v_lr,
        weight_decay=args.v_weight_decay,
        cos_loss_weight=args.v_cos_loss_weight,
        val_frac=args.v_val_frac,
        wandb_run=wandb_run,
        wandb_log_step_offset=num_train_sequences,
    )

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
        "ridge_v_weights": [w.cpu() for w in v_weights],
        "ridge_v_biases": [b.cpu() for b in v_biases],
        "v_mlp_hidden_dim": args.v_hidden_dim,
        "v_mlp_dropout": args.v_dropout,
        "v_mlp_state_dicts": [{k: v.detach().cpu() for k, v in m.state_dict().items()} for m in v_mlps],
    }
    translator_path = os.path.join(args.out_dir, "hybrid_translator.pt")
    torch.save(translator_state, translator_path)
    write_csv(v_mlp_summaries, os.path.join(args.out_dir, "v_mlp_train_summary.csv"))

    print("\nEvaluating reconstruction + next-token behavior + rollout behavior...")
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

    # Keep V MLPs on device for eval.
    v_mlps = [m.to(args.small_device).eval() for m in v_mlps]

    for eval_idx, block in enumerate(eval_iter):
        full_ids_big = block.unsqueeze(0).to(args.big_device)
        full_ids_small = block.unsqueeze(0).to(args.small_device)

        with torch.no_grad():
            big_full_out = big_model(input_ids=full_ids_big, use_cache=True)
            small_full_out = small_model(input_ids=full_ids_small, use_cache=True)
        big_full_legacy = as_legacy_cache(big_full_out.past_key_values)
        small_full_legacy = as_legacy_cache(small_full_out.past_key_values)

        ridge_full_legacy = translate_ridge_legacy_cache(
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
        hybrid_full_legacy = translate_hybrid_legacy_cache(
            big_legacy_cache=big_full_legacy,
            layer_map=layer_map,
            k_weights=k_weights,
            k_biases=k_biases,
            v_mlps=v_mlps,
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
            k_ref, v_ref = small_full_legacy[layer_idx]
            ridge_k, ridge_v = ridge_full_legacy[layer_idx]
            hybrid_k, hybrid_v = hybrid_full_legacy[layer_idx]

            yk = flatten_kv(k_ref)
            yv = flatten_kv(v_ref)
            ridge_xk = flatten_kv(ridge_k)
            ridge_xv = flatten_kv(ridge_v)
            hybrid_xv = flatten_kv(hybrid_v)

            row = {
                "eval_idx": eval_idx,
                "layer_idx": layer_idx,
                "big_layer_idx": layer_map[layer_idx],
                "ridge_k_mse": float(F.mse_loss(ridge_xk.float(), yk.float()).item()),
                "ridge_v_mse": float(F.mse_loss(ridge_xv.float(), yv.float()).item()),
                "ridge_k_cos": mean_cosine_rows(ridge_xk, yk),
                "ridge_v_cos": mean_cosine_rows(ridge_xv, yv),
                "hybrid_v_mse": float(F.mse_loss(hybrid_xv.float(), yv.float()).item()),
                "hybrid_v_cos": mean_cosine_rows(hybrid_xv, yv),
            }
            if identity_possible:
                _, v_id = identity_full_legacy[layer_idx]
                xv_id = flatten_kv(v_id)
                row.update(
                    {
                        "identity_v_mse": float(F.mse_loss(xv_id.float(), yv.float()).item()),
                        "identity_v_cos": mean_cosine_rows(xv_id, yv),
                    }
                )
            recon_rows.append(row)

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

        ridge_context_legacy = translate_ridge_legacy_cache(
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
        hybrid_context_legacy = translate_hybrid_legacy_cache(
            big_legacy_cache=big_context_legacy,
            layer_map=layer_map,
            k_weights=k_weights,
            k_biases=k_biases,
            v_mlps=v_mlps,
            small_num_kv_heads=small_num_kv_heads,
            small_head_dim=small_head_dim,
            out_device=args.small_device,
            out_dtype=small_model_dtype,
        )

        all_layers = list(range(num_small_layers))
        eval_caches = {
            "ridge": ridge_context_legacy,
            "hybrid": hybrid_context_legacy,
            "konly": merge_legacy_caches(
                native_small_legacy=small_context_legacy,
                translated_small_legacy=ridge_context_legacy,
                translated_k_layers=all_layers,
                translated_v_layers=[],
            ),
            "mlpvonly": merge_legacy_caches(
                native_small_legacy=small_context_legacy,
                translated_small_legacy=hybrid_context_legacy,
                translated_k_layers=[],
                translated_v_layers=all_layers,
            ),
        }
        if identity_possible:
            eval_caches["identity"] = identity_translate_legacy_cache(
                big_legacy_cache=big_context_legacy,
                layer_map=layer_map,
                small_num_kv_heads=small_num_kv_heads,
                small_head_dim=small_head_dim,
                out_device=args.small_device,
                out_dtype=small_model_dtype,
            )

        eval_outputs = {}
        for mode_name, mode_legacy in eval_caches.items():
            mode_cache = legacy_to_cache(mode_legacy)
            with torch.no_grad():
                eval_outputs[mode_name] = small_model(
                    input_ids=current_small,
                    past_key_values=mode_cache,
                    use_cache=True,
                )

        big_logits = big_next_out.logits[:, -1, :].to(args.small_device)
        small_native_logits = small_native_next_out.logits[:, -1, :]
        ridge_logits = eval_outputs["ridge"].logits[:, -1, :]
        hybrid_logits = eval_outputs["hybrid"].logits[:, -1, :]
        konly_logits = eval_outputs["konly"].logits[:, -1, :]
        mlpvonly_logits = eval_outputs["mlpvonly"].logits[:, -1, :]

        next_row = {
            "eval_idx": eval_idx,
            **metric_prefix_dict("native_vs_big", big_logits, small_native_logits, topk=args.topk),
            **metric_prefix_dict("ridge_vs_big", big_logits, ridge_logits, topk=args.topk),
            **metric_prefix_dict("ridge_vs_native", small_native_logits, ridge_logits, topk=args.topk),
            **metric_prefix_dict("hybrid_vs_big", big_logits, hybrid_logits, topk=args.topk),
            **metric_prefix_dict("hybrid_vs_native", small_native_logits, hybrid_logits, topk=args.topk),
            **metric_prefix_dict("konly_vs_big", big_logits, konly_logits, topk=args.topk),
            **metric_prefix_dict("konly_vs_native", small_native_logits, konly_logits, topk=args.topk),
            **metric_prefix_dict("mlpvonly_vs_big", big_logits, mlpvonly_logits, topk=args.topk),
            **metric_prefix_dict("mlpvonly_vs_native", small_native_logits, mlpvonly_logits, topk=args.topk),
        }

        if identity_possible:
            identity_logits = eval_outputs["identity"].logits[:, -1, :]
            next_row.update(metric_prefix_dict("identity_vs_big", big_logits, identity_logits, topk=args.topk))


        next_token_rows.append(next_row)

        if eval_idx % 10 == 0:
            print(f"  eval sequence {eval_idx + 1} / {args.eval_sequences}")

    recon_summary = aggregate_metric_rows(recon_rows, exclude=["eval_idx", "layer_idx", "big_layer_idx"])
    next_summary = aggregate_metric_rows(next_token_rows, exclude=["eval_idx"])

    if wandb_run is not None:
        eval_log: Dict[str, float] = {}
        for k, v in recon_summary.items():
            if isinstance(v, (int, float)) and v == v:
                wandb_run.summary[f"eval/recon/{k}"] = v
                eval_log[f"eval/recon/{k}"] = float(v)
        for k, v in next_summary.items():
            if isinstance(v, (int, float)) and v == v:
                wandb_run.summary[f"eval/next_token/{k}"] = v
                eval_log[f"eval/next_token/{k}"] = float(v)
        for row in v_mlp_summaries:
            li = int(row["layer_idx"])
            for k, v in row.items():
                if k == "layer_idx":
                    continue
                if isinstance(v, (int, float)) and v == v:
                    wandb_run.summary[f"v_mlp/final/layer_{li}/{k}"] = v

        eval_step = num_train_sequences + num_small_layers * args.v_epochs
        if eval_log:
            wandb_run.log(eval_log, step=eval_step)

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
    print(f"  {os.path.join(args.out_dir, 'v_mlp_train_summary.csv')}")
    print(f"  {os.path.join(args.out_dir, 'summary.json')}")
    print(f"  {os.path.join(args.out_dir, 'reconstruction_rows.csv')}")
    print(f"  {os.path.join(args.out_dir, 'reconstruction_per_layer.csv')}")
    print(f"  {os.path.join(args.out_dir, 'next_token_rows.csv')}")

    print("\nHigh-level summary:")
    print(result)


if __name__ == "__main__":
    main()
