#!/usr/bin/env python3
import argparse
import atexit
import csv
import math
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from kv_utils import (
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


def write_csv(rows: List[Dict], path: str) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean_cosine_rows(x: torch.Tensor, y: torch.Tensor) -> float:
    return float(F.cosine_similarity(x.float(), y.float(), dim=-1).mean().item())


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
    metrics = distribution_metrics(ref_logits, test_logits, topk=topk)
    out = {
        f"{prefix}_top1_match": float(
            (test_logits.argmax(dim=-1) == ref_logits.argmax(dim=-1)).float().mean().item()
        )
    }
    out.update({f"{prefix}_{k}": v for k, v in metrics.items()})
    return out


def parse_target_spec(target_spec: str) -> Tuple[bool, bool]:
    aliases = {
        "k": "keys",
        "key": "keys",
        "keys": "keys",
        "v": "values",
        "value": "values",
        "values": "values",
        "both": "both",
    }
    raw_items = [part.strip().lower() for part in target_spec.split(",") if part.strip()]
    if not raw_items:
        raise ValueError("--train_targets cannot be empty.")
    normalized = {aliases.get(item, item) for item in raw_items}
    if "both" in normalized:
        normalized.update({"keys", "values"})
        normalized.discard("both")
    invalid = normalized.difference({"keys", "values"})
    if invalid:
        raise ValueError(f"Unsupported target(s): {sorted(invalid)}. Use keys, values, or both.")
    return "keys" in normalized, "values" in normalized


def current_lr(optimizer: torch.optim.Optimizer) -> float:
    if not optimizer.param_groups:
        return 0.0
    return float(optimizer.param_groups[0]["lr"])


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_optimizer_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
) -> Optional[torch.optim.lr_scheduler.LambdaLR]:
    if total_optimizer_steps <= 0:
        return None

    warmup_steps = max(0, min(warmup_steps, total_optimizer_steps))

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        if total_optimizer_steps <= warmup_steps:
            return 1.0
        progress = (step - warmup_steps) / float(max(1, total_optimizer_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def sample_row_pairs(
    x: torch.Tensor,
    y: torch.Tensor,
    position_stride: int,
    max_rows_per_block: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    x = x[::position_stride]
    y = y[::position_stride]
    if max_rows_per_block > 0 and x.shape[0] > max_rows_per_block:
        idx_cpu = torch.randperm(x.shape[0])[:max_rows_per_block]
        x = x.index_select(0, idx_cpu.to(device=x.device))
        y = y.index_select(0, idx_cpu.to(device=y.device))
    return x, y


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


class LowRankResidualTranslator(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        rank: int,
        dropout: float = 0.0,
        use_layernorm: bool = False,
        use_bias: bool = False,
        residual: bool = True,
    ):
        super().__init__()
        if rank < 1:
            raise ValueError("rank must be >= 1")
        self.d_in = d_in
        self.d_out = d_out
        self.rank = rank
        self.use_residual = residual and d_in == d_out
        self.norm = nn.LayerNorm(d_in) if use_layernorm else nn.Identity()
        self.down = nn.Linear(d_in, rank, bias=use_bias)
        self.up = nn.Linear(rank, d_out, bias=use_bias)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.down.weight)
        if self.down.bias is not None:
            nn.init.zeros_(self.down.bias)

        if self.use_residual:
            nn.init.zeros_(self.up.weight)
            if self.up.bias is not None:
                nn.init.zeros_(self.up.bias)
        else:
            nn.init.xavier_uniform_(self.up.weight)
            if self.up.bias is not None:
                nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.use_residual else None
        h = self.norm(x)
        h = self.down(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.up(h)
        if residual is not None:
            h = h + residual
        return h


class FactorizedKVTranslator(nn.Module):
    def __init__(
        self,
        num_layers: int,
        d_in: int,
        d_out: int,
        k_rank: Optional[int],
        v_rank: Optional[int],
        dropout: float,
        use_layernorm: bool,
        use_bias: bool,
        residual: bool,
        train_keys: bool,
        train_values: bool,
    ):
        super().__init__()
        self.train_keys = train_keys
        self.train_values = train_values
        self.k_modules = nn.ModuleList(
            [
                LowRankResidualTranslator(
                    d_in=d_in,
                    d_out=d_out,
                    rank=int(k_rank),
                    dropout=dropout,
                    use_layernorm=use_layernorm,
                    use_bias=use_bias,
                    residual=residual,
                )
                for _ in range(num_layers)
            ]
        ) if train_keys else nn.ModuleList()
        self.v_modules = nn.ModuleList(
            [
                LowRankResidualTranslator(
                    d_in=d_in,
                    d_out=d_out,
                    rank=int(v_rank),
                    dropout=dropout,
                    use_layernorm=use_layernorm,
                    use_bias=use_bias,
                    residual=residual,
                )
                for _ in range(num_layers)
            ]
        ) if train_values else nn.ModuleList()

    def module_device(self) -> torch.device:
        return next(self.parameters()).device

    def parameter_rows(self, layer_map: List[int]) -> List[Dict[str, int]]:
        rows = []
        for layer_idx in range(len(layer_map)):
            row = {
                "layer_idx": int(layer_idx),
                "big_layer_idx": int(layer_map[layer_idx]),
                "k_params": int(sum(p.numel() for p in self.k_modules[layer_idx].parameters())) if self.train_keys else 0,
                "v_params": int(sum(p.numel() for p in self.v_modules[layer_idx].parameters())) if self.train_values else 0,
            }
            row["total_params"] = int(row["k_params"] + row["v_params"])
            rows.append(row)
        return rows

    @torch.no_grad()
    def translate_legacy_cache(
        self,
        big_legacy_cache,
        native_small_legacy,
        layer_map: List[int],
        small_num_kv_heads: int,
        small_head_dim: int,
        out_dtype: torch.dtype,
        translate_keys: Optional[bool] = None,
        translate_values: Optional[bool] = None,
    ):
        use_keys = self.train_keys if translate_keys is None else translate_keys
        use_values = self.train_values if translate_values is None else translate_values
        device = self.module_device()
        translated = []

        for small_layer_idx, big_layer_idx in enumerate(layer_map):
            k_big, v_big = big_legacy_cache[big_layer_idx]
            k_native, v_native = native_small_legacy[small_layer_idx]
            bsz, _, seqlen, _ = k_big.shape

            if use_keys:
                if not self.train_keys:
                    raise ValueError("Requested key translation, but key translators were not created.")
                xk = flatten_kv(k_big).to(device=device, dtype=torch.float32)
                yk = self.k_modules[small_layer_idx](xk)
                k_out = unflatten_kv(yk.to(dtype=out_dtype), bsz, seqlen, small_num_kv_heads, small_head_dim)
            else:
                k_out = k_native.to(device=device, dtype=out_dtype)

            if use_values:
                if not self.train_values:
                    raise ValueError("Requested value translation, but value translators were not created.")
                xv = flatten_kv(v_big).to(device=device, dtype=torch.float32)
                yv = self.v_modules[small_layer_idx](xv)
                v_out = unflatten_kv(yv.to(dtype=out_dtype), bsz, seqlen, small_num_kv_heads, small_head_dim)
            else:
                v_out = v_native.to(device=device, dtype=out_dtype)

            translated.append((k_out, v_out))
        return tuple(translated)


def identity_translate_legacy_cache(
    big_legacy_cache,
    native_small_legacy,
    layer_map: List[int],
    small_num_kv_heads: int,
    small_head_dim: int,
    out_device: torch.device,
    out_dtype: torch.dtype,
    translate_keys: bool,
    translate_values: bool,
):
    translated = []
    for small_layer_idx, big_layer_idx in enumerate(layer_map):
        k_big, v_big = big_legacy_cache[big_layer_idx]
        k_native, v_native = native_small_legacy[small_layer_idx]
        bsz, _, seqlen, _ = k_big.shape

        if translate_keys:
            xk = flatten_kv(k_big).to(device=out_device)
            k_out = unflatten_kv(xk.to(dtype=out_dtype), bsz, seqlen, small_num_kv_heads, small_head_dim)
        else:
            k_out = k_native.to(device=out_device, dtype=out_dtype)

        if translate_values:
            xv = flatten_kv(v_big).to(device=out_device)
            v_out = unflatten_kv(xv.to(dtype=out_dtype), bsz, seqlen, small_num_kv_heads, small_head_dim)
        else:
            v_out = v_native.to(device=out_device, dtype=out_dtype)

        translated.append((k_out, v_out))
    return tuple(translated)


def reconstruction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    cos_loss_weight: float,
) -> Tuple[torch.Tensor, float, float]:
    mse = F.mse_loss(pred, target)
    cos_sim = F.cosine_similarity(pred, target, dim=-1).mean()
    loss = mse + cos_loss_weight * (1.0 - cos_sim)
    return loss, float(mse.item()), float(cos_sim.item())


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


def log_wandb_table(wandb_run: Optional[Any], key: str, rows: List[Dict]) -> None:
    if wandb_run is None or not rows:
        return
    try:
        import wandb
    except ImportError:
        return

    columns = list(rows[0].keys())
    data = [[row.get(column) for column in columns] for row in rows]
    wandb_run.log({key: wandb.Table(columns=columns, data=data)})


def log_wandb_artifact(
    wandb_run: Optional[Any],
    artifact_name: str,
    file_paths: List[str],
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    if wandb_run is None or not file_paths:
        return
    try:
        import wandb
    except ImportError:
        return

    artifact = wandb.Artifact(name=artifact_name, type="kv-factorized-run", metadata=metadata or {})
    for path in file_paths:
        if os.path.isfile(path):
            artifact.add_file(path)
    wandb_run.log_artifact(artifact)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a streamed low-rank neural KV translator from a big model cache into a small model cache."
    )
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
    parser.add_argument("--stream_train", dest="stream_train", action="store_true")
    parser.add_argument("--no_stream_train", dest="stream_train", action="store_false")
    parser.set_defaults(stream_train=True)
    parser.add_argument("--stream_eval", action="store_true")
    parser.add_argument("--shuffle_train", action="store_true")
    parser.add_argument("--shuffle_buffer_size", type=int, default=10_000)

    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--train_sequences", type=int, default=4096)
    parser.add_argument("--eval_sequences", type=int, default=128)
    parser.add_argument("--position_stride", type=int, default=1)
    parser.add_argument("--max_rows_per_layer_per_block", type=int, default=128)
    parser.add_argument("--layer_map", type=str, choices=["depth"], default="depth")
    parser.add_argument("--train_targets", type=str, default="both")

    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--k_rank", type=int, default=None)
    parser.add_argument("--v_rank", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--use_layernorm", action="store_true")
    parser.add_argument("--use_bias", action="store_true")
    parser.add_argument("--no_residual", action="store_true")

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--cos_loss_weight", type=float, default=0.1)
    parser.add_argument("--grad_accum_sequences", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow_incompatible_tokenizers", action="store_true")

    parser.add_argument("--out_dir", type=str, default="outputs/kv_factorized_probe")

    parser.add_argument("--wandb", action="store_true", help="Log training/eval metrics to Weights & Biases.")
    parser.add_argument("--wandb_project", type=str, default="kv-reduce")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.seq_len < 2:
        raise ValueError("seq_len must be at least 2")
    if args.train_sequences < 1:
        raise ValueError("train_sequences must be >= 1")
    if args.eval_sequences < 1:
        raise ValueError("eval_sequences must be >= 1")
    if args.position_stride < 1:
        raise ValueError("position_stride must be >= 1")
    if args.grad_accum_sequences < 1:
        raise ValueError("grad_accum_sequences must be >= 1")
    if args.shuffle_buffer_size < 1:
        raise ValueError("shuffle_buffer_size must be >= 1")

    train_keys, train_values = parse_target_spec(args.train_targets)
    k_rank = args.k_rank if args.k_rank is not None else args.rank
    v_rank = args.v_rank if args.v_rank is not None else args.rank

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    wandb_run = init_wandb(args)

    print("Loading models and tokenizers...")
    big_tokenizer = load_tokenizer(args.big_model)
    small_tokenizer = load_tokenizer(args.small_model)
    compatibility = tokenizer_compatibility_report(big_tokenizer, small_tokenizer)
    if (not compatibility["all_probe_encodings_match"]) and (not args.allow_incompatible_tokenizers):
        raise ValueError(
            "Tokenizers appear incompatible. Use the same tokenizer family or pass --allow_incompatible_tokenizers."
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
    print(f"Training targets: keys={train_keys}, values={train_values}")
    print(f"Streaming train split: {args.stream_train}")

    translator = FactorizedKVTranslator(
        num_layers=num_small_layers,
        d_in=big_kv_dim,
        d_out=small_kv_dim,
        k_rank=k_rank,
        v_rank=v_rank,
        dropout=args.dropout,
        use_layernorm=args.use_layernorm,
        use_bias=args.use_bias,
        residual=not args.no_residual,
        train_keys=train_keys,
        train_values=train_values,
    ).to(args.small_device)
    translator.train()

    parameter_rows = translator.parameter_rows(layer_map)
    parameter_summary = {
        "total_params": int(sum(row["total_params"] for row in parameter_rows)),
        "k_total_params": int(sum(row["k_params"] for row in parameter_rows)),
        "v_total_params": int(sum(row["v_params"] for row in parameter_rows)),
    }

    if wandb_run is not None:
        for key, value in parameter_summary.items():
            wandb_run.summary[f"params/{key}"] = value
        wandb_run.summary["model/big_num_layers"] = num_big_layers
        wandb_run.summary["model/small_num_layers"] = num_small_layers
        wandb_run.summary["model/big_kv_dim"] = big_kv_dim
        wandb_run.summary["model/small_kv_dim"] = small_kv_dim
        wandb_run.summary["data/train_streaming"] = bool(args.stream_train)
        wandb_run.summary["data/train_targets_keys"] = bool(train_keys)
        wandb_run.summary["data/train_targets_values"] = bool(train_values)

    optimizer = torch.optim.AdamW(translator.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_optimizer_steps = math.ceil(args.train_sequences / args.grad_accum_sequences)
    scheduler = build_scheduler(
        optimizer=optimizer,
        total_optimizer_steps=total_optimizer_steps,
        warmup_steps=args.warmup_steps,
        min_lr_ratio=args.min_lr_ratio,
    )

    print("\nTraining low-rank translators from streamed cache pairs...")
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
        streaming=args.stream_train,
        shuffle_buffer_size=args.shuffle_buffer_size,
    )

    optimizer.zero_grad(set_to_none=True)
    log_window = defaultdict(float)
    layer_train_stats = [defaultdict(float) for _ in range(num_small_layers)]
    train_log_rows = []
    pending_updates = 0
    optimizer_step = 0
    train_idx_last = -1

    train_bar = tqdm(train_iter, total=args.train_sequences, desc="Train blocks", unit="block")
    for train_idx, block in enumerate(train_bar):
        train_idx_last = train_idx
        full_ids_big = block.unsqueeze(0).to(args.big_device)
        full_ids_small = block.unsqueeze(0).to(args.small_device)

        with torch.no_grad():
            big_out = big_model(input_ids=full_ids_big, use_cache=True)
            small_out = small_model(input_ids=full_ids_small, use_cache=True)
        big_legacy = as_legacy_cache(big_out.past_key_values)
        small_legacy = as_legacy_cache(small_out.past_key_values)

        loss_terms = []
        step_metrics = defaultdict(float)

        for small_layer_idx, big_layer_idx in enumerate(layer_map):
            k_big, v_big = big_legacy[big_layer_idx]
            k_small, v_small = small_legacy[small_layer_idx]

            if train_keys:
                xk, yk = sample_row_pairs(
                    flatten_kv(k_big).float(),
                    flatten_kv(k_small).float(),
                    position_stride=args.position_stride,
                    max_rows_per_block=args.max_rows_per_layer_per_block,
                )
                xk = xk.to(device=args.small_device, dtype=torch.float32, non_blocking=True)
                yk = yk.to(device=args.small_device, dtype=torch.float32, non_blocking=True)
                pred_k = translator.k_modules[small_layer_idx](xk)
                loss_k, mse_k, cos_k = reconstruction_loss(pred_k, yk, args.cos_loss_weight)
                loss_terms.append(loss_k)
                step_metrics["k_loss_sum"] += float(loss_k.item())
                step_metrics["k_mse_sum"] += mse_k
                step_metrics["k_cos_sum"] += cos_k
                step_metrics["k_rows_sum"] += int(xk.shape[0])
                step_metrics["k_terms"] += 1

                layer_train_stats[small_layer_idx]["k_loss_sum"] += float(loss_k.item())
                layer_train_stats[small_layer_idx]["k_mse_sum"] += mse_k
                layer_train_stats[small_layer_idx]["k_cos_sum"] += cos_k
                layer_train_stats[small_layer_idx]["k_rows_sum"] += int(xk.shape[0])
                layer_train_stats[small_layer_idx]["k_terms"] += 1

            if train_values:
                xv, yv = sample_row_pairs(
                    flatten_kv(v_big).float(),
                    flatten_kv(v_small).float(),
                    position_stride=args.position_stride,
                    max_rows_per_block=args.max_rows_per_layer_per_block,
                )
                xv = xv.to(device=args.small_device, dtype=torch.float32, non_blocking=True)
                yv = yv.to(device=args.small_device, dtype=torch.float32, non_blocking=True)
                pred_v = translator.v_modules[small_layer_idx](xv)
                loss_v, mse_v, cos_v = reconstruction_loss(pred_v, yv, args.cos_loss_weight)
                loss_terms.append(loss_v)
                step_metrics["v_loss_sum"] += float(loss_v.item())
                step_metrics["v_mse_sum"] += mse_v
                step_metrics["v_cos_sum"] += cos_v
                step_metrics["v_rows_sum"] += int(xv.shape[0])
                step_metrics["v_terms"] += 1

                layer_train_stats[small_layer_idx]["v_loss_sum"] += float(loss_v.item())
                layer_train_stats[small_layer_idx]["v_mse_sum"] += mse_v
                layer_train_stats[small_layer_idx]["v_cos_sum"] += cos_v
                layer_train_stats[small_layer_idx]["v_rows_sum"] += int(xv.shape[0])
                layer_train_stats[small_layer_idx]["v_terms"] += 1

        if not loss_terms:
            raise ValueError("No train targets were enabled.")

        step_loss = torch.stack(loss_terms).mean()
        (step_loss / args.grad_accum_sequences).backward()
        pending_updates += 1

        step_metrics["total_loss"] = float(step_loss.item())
        step_metrics["num_terms"] = float(len(loss_terms))
        for key, value in step_metrics.items():
            log_window[key] += float(value)
        log_window["num_sequences"] += 1.0

        postfix = {"loss": f"{step_loss.item():.5f}"}
        if train_keys and step_metrics["k_terms"] > 0:
            postfix["k_cos"] = f"{step_metrics['k_cos_sum'] / step_metrics['k_terms']:.4f}"
        if train_values and step_metrics["v_terms"] > 0:
            postfix["v_cos"] = f"{step_metrics['v_cos_sum'] / step_metrics['v_terms']:.4f}"
        train_bar.set_postfix(postfix, refresh=False)

        if pending_updates == args.grad_accum_sequences:
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(translator.parameters(), args.grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1
            pending_updates = 0

        should_log = ((train_idx + 1) % args.log_every == 0) or ((train_idx + 1) == args.train_sequences)
        if should_log and log_window["num_sequences"] > 0:
            window_sequences = log_window["num_sequences"]
            row = {
                "sequence_idx": int(train_idx + 1),
                "optimizer_step": int(optimizer_step),
                "lr": current_lr(optimizer),
                "avg_total_loss": float(log_window["total_loss"] / window_sequences),
            }
            if train_keys and log_window["k_terms"] > 0:
                row.update(
                    {
                        "avg_k_loss": float(log_window["k_loss_sum"] / log_window["k_terms"]),
                        "avg_k_mse": float(log_window["k_mse_sum"] / log_window["k_terms"]),
                        "avg_k_cos": float(log_window["k_cos_sum"] / log_window["k_terms"]),
                        "avg_k_rows": float(log_window["k_rows_sum"] / window_sequences),
                    }
                )
            if train_values and log_window["v_terms"] > 0:
                row.update(
                    {
                        "avg_v_loss": float(log_window["v_loss_sum"] / log_window["v_terms"]),
                        "avg_v_mse": float(log_window["v_mse_sum"] / log_window["v_terms"]),
                        "avg_v_cos": float(log_window["v_cos_sum"] / log_window["v_terms"]),
                        "avg_v_rows": float(log_window["v_rows_sum"] / window_sequences),
                    }
                )
            train_log_rows.append(row)

            if wandb_run is not None:
                wandb_log = {f"train/{k}": v for k, v in row.items()}
                wandb_run.log(wandb_log, step=train_idx + 1)

            log_window = defaultdict(float)

    train_bar.close()

    if pending_updates > 0:
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(translator.parameters(), args.grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_step += 1

    num_train_sequences = train_idx_last + 1 if train_idx_last >= 0 else 0
    if num_train_sequences == 0:
        raise ValueError("Training produced zero token blocks. Check your dataset arguments.")

    if log_window["num_sequences"] > 0:
        window_sequences = log_window["num_sequences"]
        row = {
            "sequence_idx": int(num_train_sequences),
            "optimizer_step": int(optimizer_step),
            "lr": current_lr(optimizer),
            "avg_total_loss": float(log_window["total_loss"] / window_sequences),
        }
        if train_keys and log_window["k_terms"] > 0:
            row.update(
                {
                    "avg_k_loss": float(log_window["k_loss_sum"] / log_window["k_terms"]),
                    "avg_k_mse": float(log_window["k_mse_sum"] / log_window["k_terms"]),
                    "avg_k_cos": float(log_window["k_cos_sum"] / log_window["k_terms"]),
                    "avg_k_rows": float(log_window["k_rows_sum"] / window_sequences),
                }
            )
        if train_values and log_window["v_terms"] > 0:
            row.update(
                {
                    "avg_v_loss": float(log_window["v_loss_sum"] / log_window["v_terms"]),
                    "avg_v_mse": float(log_window["v_mse_sum"] / log_window["v_terms"]),
                    "avg_v_cos": float(log_window["v_cos_sum"] / log_window["v_terms"]),
                    "avg_v_rows": float(log_window["v_rows_sum"] / window_sequences),
                }
            )
        train_log_rows.append(row)
        if wandb_run is not None:
            wandb_log = {f"train/{k}": v for k, v in row.items()}
            wandb_run.log(wandb_log, step=num_train_sequences)

    translator.eval()

    train_per_layer_rows = []
    for layer_idx, stats in enumerate(layer_train_stats):
        row = {
            "layer_idx": int(layer_idx),
            "big_layer_idx": int(layer_map[layer_idx]),
        }
        if train_keys and stats["k_terms"] > 0:
            row.update(
                {
                    "k_avg_loss": float(stats["k_loss_sum"] / stats["k_terms"]),
                    "k_avg_mse": float(stats["k_mse_sum"] / stats["k_terms"]),
                    "k_avg_cos": float(stats["k_cos_sum"] / stats["k_terms"]),
                    "k_avg_rows": float(stats["k_rows_sum"] / stats["k_terms"]),
                }
            )
        if train_values and stats["v_terms"] > 0:
            row.update(
                {
                    "v_avg_loss": float(stats["v_loss_sum"] / stats["v_terms"]),
                    "v_avg_mse": float(stats["v_mse_sum"] / stats["v_terms"]),
                    "v_avg_cos": float(stats["v_cos_sum"] / stats["v_terms"]),
                    "v_avg_rows": float(stats["v_rows_sum"] / stats["v_terms"]),
                }
            )
        train_per_layer_rows.append(row)

    translator_path = os.path.join(args.out_dir, "factorized_translator.pt")
    torch.save(
        {
            "args": vars(args),
            "big_model": args.big_model,
            "small_model": args.small_model,
            "layer_map": layer_map,
            "big_kv_dim": big_kv_dim,
            "small_kv_dim": small_kv_dim,
            "small_num_kv_heads": small_num_kv_heads,
            "small_head_dim": small_head_dim,
            "train_keys": train_keys,
            "train_values": train_values,
            "k_rank": k_rank,
            "v_rank": v_rank,
            "use_layernorm": bool(args.use_layernorm),
            "use_bias": bool(args.use_bias),
            "residual": bool(not args.no_residual),
            "parameter_summary": parameter_summary,
            "state_dict": {k: v.detach().cpu() for k, v in translator.state_dict().items()},
        },
        translator_path,
    )

    print("\nEvaluating reconstruction + next-token behavior...")
    translator = translator.to(args.small_device).eval()
    identity_possible = big_kv_dim == small_kv_dim

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
        streaming=args.stream_eval,
        shuffle_buffer_size=args.shuffle_buffer_size,
    )

    eval_bar = tqdm(eval_iter, total=args.eval_sequences, desc="Eval blocks", unit="block")
    eval_idx_last = -1
    for eval_idx, block in enumerate(eval_bar):
        eval_idx_last = eval_idx
        full_ids_big = block.unsqueeze(0).to(args.big_device)
        full_ids_small = block.unsqueeze(0).to(args.small_device)

        with torch.no_grad():
            big_full_out = big_model(input_ids=full_ids_big, use_cache=True)
            small_full_out = small_model(input_ids=full_ids_small, use_cache=True)
        big_full_legacy = as_legacy_cache(big_full_out.past_key_values)
        small_full_legacy = as_legacy_cache(small_full_out.past_key_values)
        translated_full_legacy = translator.translate_legacy_cache(
            big_legacy_cache=big_full_legacy,
            native_small_legacy=small_full_legacy,
            layer_map=layer_map,
            small_num_kv_heads=small_num_kv_heads,
            small_head_dim=small_head_dim,
            out_dtype=small_model_dtype,
        )
        identity_full_legacy = None
        if identity_possible:
            identity_full_legacy = identity_translate_legacy_cache(
                big_legacy_cache=big_full_legacy,
                native_small_legacy=small_full_legacy,
                layer_map=layer_map,
                small_num_kv_heads=small_num_kv_heads,
                small_head_dim=small_head_dim,
                out_device=translator.module_device(),
                out_dtype=small_model_dtype,
                translate_keys=train_keys,
                translate_values=train_values,
            )

        for layer_idx in range(num_small_layers):
            row = {
                "eval_idx": int(eval_idx),
                "layer_idx": int(layer_idx),
                "big_layer_idx": int(layer_map[layer_idx]),
            }
            k_ref, v_ref = small_full_legacy[layer_idx]
            k_hat, v_hat = translated_full_legacy[layer_idx]

            if train_keys:
                xk = flatten_kv(k_hat)
                yk = flatten_kv(k_ref)
                row["k_mse"] = float(F.mse_loss(xk.float(), yk.float()).item())
                row["k_cos"] = mean_cosine_rows(xk, yk)
                if identity_possible:
                    k_id, _ = identity_full_legacy[layer_idx]
                    xk_id = flatten_kv(k_id)
                    row["k_mse_identity"] = float(F.mse_loss(xk_id.float(), yk.float()).item())
                    row["k_cos_identity"] = mean_cosine_rows(xk_id, yk)

            if train_values:
                xv = flatten_kv(v_hat)
                yv = flatten_kv(v_ref)
                row["v_mse"] = float(F.mse_loss(xv.float(), yv.float()).item())
                row["v_cos"] = mean_cosine_rows(xv, yv)
                if identity_possible:
                    _, v_id = identity_full_legacy[layer_idx]
                    xv_id = flatten_kv(v_id)
                    row["v_mse_identity"] = float(F.mse_loss(xv_id.float(), yv.float()).item())
                    row["v_cos_identity"] = mean_cosine_rows(xv_id, yv)

            recon_rows.append(row)

        context_big = full_ids_big[:, :-1]
        context_small = full_ids_small[:, :-1]
        current_big = full_ids_big[:, -1:]
        current_small = full_ids_small[:, -1:]

        with torch.no_grad():
            big_context_out = big_model(input_ids=context_big, use_cache=True)
            small_context_out = small_model(input_ids=context_small, use_cache=True)
            big_next_out = big_model(
                input_ids=current_big,
                past_key_values=big_context_out.past_key_values,
                use_cache=True,
            )
            small_native_next_out = small_model(
                input_ids=current_small,
                past_key_values=small_context_out.past_key_values,
                use_cache=True,
            )

        big_context_legacy = as_legacy_cache(big_context_out.past_key_values)
        small_context_legacy = as_legacy_cache(small_context_out.past_key_values)

        translated_context_legacy = translator.translate_legacy_cache(
            big_legacy_cache=big_context_legacy,
            native_small_legacy=small_context_legacy,
            layer_map=layer_map,
            small_num_kv_heads=small_num_kv_heads,
            small_head_dim=small_head_dim,
            out_dtype=small_model_dtype,
        )

        eval_caches = {"translated": translated_context_legacy}
        if train_keys and train_values:
            all_layers = list(range(num_small_layers))
            eval_caches["konly"] = merge_legacy_caches(
                native_small_legacy=small_context_legacy,
                translated_small_legacy=translated_context_legacy,
                translated_k_layers=all_layers,
                translated_v_layers=[],
            )
            eval_caches["vonly"] = merge_legacy_caches(
                native_small_legacy=small_context_legacy,
                translated_small_legacy=translated_context_legacy,
                translated_k_layers=[],
                translated_v_layers=all_layers,
            )

        if identity_possible:
            eval_caches["identity"] = identity_translate_legacy_cache(
                big_legacy_cache=big_context_legacy,
                native_small_legacy=small_context_legacy,
                layer_map=layer_map,
                small_num_kv_heads=small_num_kv_heads,
                small_head_dim=small_head_dim,
                out_device=translator.module_device(),
                out_dtype=small_model_dtype,
                translate_keys=train_keys,
                translate_values=train_values,
            )

        eval_outputs = {}
        for mode_name, mode_legacy in eval_caches.items():
            with torch.no_grad():
                eval_outputs[mode_name] = small_model(
                    input_ids=current_small,
                    past_key_values=legacy_to_cache(mode_legacy),
                    use_cache=True,
                )

        big_logits = big_next_out.logits[:, -1, :].to(args.small_device)
        small_native_logits = small_native_next_out.logits[:, -1, :]
        translated_logits = eval_outputs["translated"].logits[:, -1, :]

        next_row = {
            "eval_idx": int(eval_idx),
            **metric_prefix_dict("native_vs_big", big_logits, small_native_logits, topk=args.topk),
            **metric_prefix_dict("translated_vs_big", big_logits, translated_logits, topk=args.topk),
            **metric_prefix_dict("translated_vs_native", small_native_logits, translated_logits, topk=args.topk),
        }
        if "konly" in eval_outputs:
            konly_logits = eval_outputs["konly"].logits[:, -1, :]
            next_row.update(metric_prefix_dict("konly_vs_big", big_logits, konly_logits, topk=args.topk))
            next_row.update(metric_prefix_dict("konly_vs_native", small_native_logits, konly_logits, topk=args.topk))
        if "vonly" in eval_outputs:
            vonly_logits = eval_outputs["vonly"].logits[:, -1, :]
            next_row.update(metric_prefix_dict("vonly_vs_big", big_logits, vonly_logits, topk=args.topk))
            next_row.update(metric_prefix_dict("vonly_vs_native", small_native_logits, vonly_logits, topk=args.topk))
        if "identity" in eval_outputs:
            identity_logits = eval_outputs["identity"].logits[:, -1, :]
            next_row.update(metric_prefix_dict("identity_vs_big", big_logits, identity_logits, topk=args.topk))
            next_row.update(metric_prefix_dict("identity_vs_native", small_native_logits, identity_logits, topk=args.topk))

        next_token_rows.append(next_row)

    eval_bar.close()

    num_eval_sequences = eval_idx_last + 1 if eval_idx_last >= 0 else 0
    if num_eval_sequences == 0:
        raise ValueError("Evaluation produced zero token blocks. Check your eval split arguments.")

    recon_summary = aggregate_metric_rows(recon_rows, exclude=["eval_idx", "layer_idx", "big_layer_idx"])
    next_summary = aggregate_metric_rows(next_token_rows, exclude=["eval_idx"])

    reconstruction_per_layer = []
    per_layer_groups = defaultdict(list)
    for row in recon_rows:
        per_layer_groups[int(row["layer_idx"])].append(row)
    for layer_idx in sorted(per_layer_groups.keys()):
        summary = aggregate_metric_rows(
            per_layer_groups[layer_idx],
            exclude=["eval_idx", "layer_idx", "big_layer_idx"],
        )
        summary["layer_idx"] = int(layer_idx)
        summary["big_layer_idx"] = int(layer_map[layer_idx])
        reconstruction_per_layer.append(summary)

    summary = {
        "args": vars(args),
        "tokenizer_compatibility": compatibility,
        "layer_map": layer_map,
        "big_model": args.big_model,
        "small_model": args.small_model,
        "big_num_layers": num_big_layers,
        "small_num_layers": num_small_layers,
        "big_kv_dim": big_kv_dim,
        "small_kv_dim": small_kv_dim,
        "train_targets": {
            "keys": bool(train_keys),
            "values": bool(train_values),
        },
        "parameter_summary": parameter_summary,
        "num_train_sequences": int(num_train_sequences),
        "num_eval_sequences": int(num_eval_sequences),
        "optimizer_steps": int(optimizer_step),
        "translator_path": translator_path,
        "identity_possible": bool(identity_possible),
        "reconstruction_summary": recon_summary,
        "next_token_summary": next_summary,
    }

    write_csv(parameter_rows, os.path.join(args.out_dir, "parameter_summary.csv"))
    write_csv(train_log_rows, os.path.join(args.out_dir, "train_log.csv"))
    write_csv(train_per_layer_rows, os.path.join(args.out_dir, "train_per_layer.csv"))
    write_csv(recon_rows, os.path.join(args.out_dir, "reconstruction_rows.csv"))
    write_csv(reconstruction_per_layer, os.path.join(args.out_dir, "reconstruction_per_layer.csv"))
    write_csv(next_token_rows, os.path.join(args.out_dir, "next_token_rows.csv"))
    write_json(summary, os.path.join(args.out_dir, "summary.json"))

    if wandb_run is not None:
        for key, value in recon_summary.items():
            if value == value:
                wandb_run.summary[f"eval/reconstruction/{key}"] = value
        for key, value in next_summary.items():
            if value == value:
                wandb_run.summary[f"eval/next_token/{key}"] = value
        for key, value in parameter_summary.items():
            wandb_run.summary[f"params/{key}"] = value
        wandb_run.summary["data/num_train_sequences"] = int(num_train_sequences)
        wandb_run.summary["data/num_eval_sequences"] = int(num_eval_sequences)
        wandb_run.summary["optimization/optimizer_steps"] = int(optimizer_step)
        wandb_run.summary["artifacts/out_dir"] = args.out_dir

        for row in train_per_layer_rows:
            layer_idx = int(row["layer_idx"])
            for key, value in row.items():
                if key in {"layer_idx", "big_layer_idx"}:
                    continue
                wandb_run.summary[f"train_per_layer/layer_{layer_idx}/{key}"] = value

        for row in reconstruction_per_layer:
            layer_idx = int(row["layer_idx"])
            for key, value in row.items():
                if key in {"layer_idx", "big_layer_idx"}:
                    continue
                wandb_run.summary[f"eval_per_layer/layer_{layer_idx}/{key}"] = value

        log_wandb_table(wandb_run, "tables/parameter_summary", parameter_rows)
        log_wandb_table(wandb_run, "tables/train_per_layer", train_per_layer_rows)
        log_wandb_table(wandb_run, "tables/reconstruction_per_layer", reconstruction_per_layer)
        log_wandb_table(wandb_run, "tables/next_token_rows", next_token_rows)

        log_wandb_artifact(
            wandb_run=wandb_run,
            artifact_name=f"kv-factorized-{wandb_run.id}",
            file_paths=[
                translator_path,
                os.path.join(args.out_dir, "parameter_summary.csv"),
                os.path.join(args.out_dir, "train_log.csv"),
                os.path.join(args.out_dir, "train_per_layer.csv"),
                os.path.join(args.out_dir, "reconstruction_rows.csv"),
                os.path.join(args.out_dir, "reconstruction_per_layer.csv"),
                os.path.join(args.out_dir, "next_token_rows.csv"),
                os.path.join(args.out_dir, "summary.json"),
            ],
            metadata={
                "big_model": args.big_model,
                "small_model": args.small_model,
                "num_train_sequences": int(num_train_sequences),
                "num_eval_sequences": int(num_eval_sequences),
                "train_targets": {"keys": bool(train_keys), "values": bool(train_values)},
                "parameter_summary": parameter_summary,
            },
        )

    print("\nSaved:")
    print(f"  {translator_path}")
    print(f"  {os.path.join(args.out_dir, 'parameter_summary.csv')}")
    print(f"  {os.path.join(args.out_dir, 'train_log.csv')}")
    print(f"  {os.path.join(args.out_dir, 'train_per_layer.csv')}")
    print(f"  {os.path.join(args.out_dir, 'reconstruction_rows.csv')}")
    print(f"  {os.path.join(args.out_dir, 'reconstruction_per_layer.csv')}")
    print(f"  {os.path.join(args.out_dir, 'next_token_rows.csv')}")
    print(f"  {os.path.join(args.out_dir, 'summary.json')}")

    print("\nHigh-level summary:")
    print(summary)


if __name__ == "__main__":
    main()
