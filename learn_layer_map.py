#!/usr/bin/env python3
"""
Learn a monotonic target-layer -> draft-layer alignment map.

This is the first revamp diagnostic for KV Reduce: instead of assuming draft layer i maps to a
linearly spaced target layer, collect content-side representations and choose a monotonic mapping
that maximizes representational similarity. Hooks intentionally collect pre-RoPE K projections so
positional rotation is not mixed into the layer alignment score.
"""

import argparse
import atexit
import csv
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from kv_utils import (
    depth_layer_map,
    iter_token_blocks,
    load_causal_lm,
    load_tokenizer,
    set_seed,
    tokenizer_compatibility_report,
    write_json,
)


DEFAULT_FEATURE_WEIGHTS = {
    "k_pre": 1.0,
    "v_pre": 0.7,
    "attn_out": 1.0,
    "residual": 0.8,
}


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


def parse_csv_items(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_feature_weights(features: Sequence[str], value: Optional[str]) -> Dict[str, float]:
    weights = {feature: DEFAULT_FEATURE_WEIGHTS.get(feature, 1.0) for feature in features}
    if value is None:
        return weights
    for item in parse_csv_items(value):
        if "=" not in item:
            raise ValueError("--feature_weights entries must look like k_pre=1.0")
        name, raw_weight = item.split("=", 1)
        name = name.strip()
        if name not in weights:
            raise ValueError(f"Feature weight {name!r} is not in selected features: {features}")
        weights[name] = float(raw_weight)
    return weights


def write_csv(rows: List[Dict[str, Any]], path: str) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_feature(
    store: Dict[int, Dict[str, List[torch.Tensor]]],
    *,
    layer_idx: int,
    feature_name: str,
    tensor: torch.Tensor,
    position_stride: int,
    max_rows: int,
) -> None:
    existing = sum(part.shape[0] for part in store[layer_idx][feature_name])
    remaining = max_rows - existing
    if remaining <= 0:
        return
    flat = tensor.detach().float().reshape(-1, tensor.shape[-1])
    if position_stride > 1:
        flat = flat[::position_stride]
    if flat.shape[0] > remaining:
        flat = flat[:remaining]
    store[layer_idx][feature_name].append(flat.cpu().to(torch.float16))


def make_attention_hook(
    *,
    layer_idx: int,
    features: Sequence[str],
    store: Dict[int, Dict[str, List[torch.Tensor]]],
    position_stride: int,
    max_rows: int,
):
    feature_set = set(features)

    def hook(module, args, kwargs, output):
        hidden_states = kwargs.get("hidden_states")
        if hidden_states is None and args:
            hidden_states = args[0]
        if hidden_states is None:
            return

        if "residual" in feature_set:
            append_feature(
                store,
                layer_idx=layer_idx,
                feature_name="residual",
                tensor=hidden_states,
                position_stride=position_stride,
                max_rows=max_rows,
            )
        if "k_pre" in feature_set:
            k_pre = module.k_proj(hidden_states)
            append_feature(
                store,
                layer_idx=layer_idx,
                feature_name="k_pre",
                tensor=k_pre,
                position_stride=position_stride,
                max_rows=max_rows,
            )
        if "v_pre" in feature_set:
            v_pre = module.v_proj(hidden_states)
            append_feature(
                store,
                layer_idx=layer_idx,
                feature_name="v_pre",
                tensor=v_pre,
                position_stride=position_stride,
                max_rows=max_rows,
            )
        if "attn_out" in feature_set:
            attn_output = output[0] if isinstance(output, tuple) else output
            append_feature(
                store,
                layer_idx=layer_idx,
                feature_name="attn_out",
                tensor=attn_output,
                position_stride=position_stride,
                max_rows=max_rows,
            )

    return hook


def materialize_features(
    store: Dict[int, Dict[str, List[torch.Tensor]]],
    *,
    num_layers: int,
    features: Sequence[str],
) -> Dict[str, List[torch.Tensor]]:
    out: Dict[str, List[torch.Tensor]] = {feature: [] for feature in features}
    for feature in features:
        for layer_idx in range(num_layers):
            parts = store[layer_idx][feature]
            if not parts:
                raise RuntimeError(f"No rows collected for layer {layer_idx}, feature {feature}.")
            out[feature].append(torch.cat(parts, dim=0).float())
    return out


def centered_gram(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    x = x - x.mean(dim=0, keepdim=True)
    return x @ x.T


def gram_cka(gram_a: torch.Tensor, gram_b: torch.Tensor, eps: float = 1e-12) -> float:
    rows = min(gram_a.shape[0], gram_b.shape[0])
    a = gram_a[:rows, :rows]
    b = gram_b[:rows, :rows]
    numerator = torch.sum(a * b)
    denominator = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    return float((numerator / denominator.clamp_min(eps)).item())


def compute_similarity_matrix(
    *,
    target_features: Dict[str, List[torch.Tensor]],
    draft_features: Dict[str, List[torch.Tensor]],
    feature_weights: Dict[str, float],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    features = list(feature_weights.keys())
    num_target_layers = len(next(iter(target_features.values())))
    num_draft_layers = len(next(iter(draft_features.values())))
    per_feature: Dict[str, torch.Tensor] = {}
    combined = torch.zeros(num_draft_layers, num_target_layers, dtype=torch.float64)
    total_weight = 0.0

    for feature in features:
        weight = float(feature_weights[feature])
        if weight == 0.0:
            continue
        target_grams = [centered_gram(x) for x in target_features[feature]]
        draft_grams = [centered_gram(x) for x in draft_features[feature]]
        matrix = torch.zeros(num_draft_layers, num_target_layers, dtype=torch.float64)
        for draft_idx, draft_gram in enumerate(draft_grams):
            for target_idx, target_gram in enumerate(target_grams):
                matrix[draft_idx, target_idx] = gram_cka(draft_gram, target_gram)
        per_feature[feature] = matrix
        combined += weight * matrix
        total_weight += weight

    if total_weight <= 0.0:
        raise ValueError("At least one selected feature must have nonzero weight.")
    combined /= total_weight
    return combined, per_feature


def monotonic_layer_map(similarity: torch.Tensor, *, allow_layer_reuse: bool) -> List[int]:
    num_draft_layers, num_target_layers = similarity.shape
    if num_target_layers < num_draft_layers:
        allow_layer_reuse = True

    neg_inf = -1.0e30
    dp = torch.full((num_draft_layers, num_target_layers), neg_inf, dtype=torch.float64)
    back = torch.full((num_draft_layers, num_target_layers), -1, dtype=torch.long)
    dp[0] = similarity[0]

    for draft_idx in range(1, num_draft_layers):
        for target_idx in range(num_target_layers):
            prev_end = target_idx + 1 if allow_layer_reuse else target_idx
            if prev_end <= 0:
                continue
            prev_scores = dp[draft_idx - 1, :prev_end]
            prev_best = int(torch.argmax(prev_scores).item())
            dp[draft_idx, target_idx] = prev_scores[prev_best] + similarity[draft_idx, target_idx]
            back[draft_idx, target_idx] = prev_best

    last = int(torch.argmax(dp[-1]).item())
    layer_map = [last]
    for draft_idx in range(num_draft_layers - 1, 0, -1):
        last = int(back[draft_idx, last].item())
        if last < 0:
            raise RuntimeError("Failed to backtrack monotonic layer map.")
        layer_map.append(last)
    layer_map.reverse()
    return layer_map


def matrix_to_rows(matrix: torch.Tensor, prefix: str = "similarity") -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for draft_idx in range(matrix.shape[0]):
        row: Dict[str, Any] = {"draft_layer": int(draft_idx)}
        for target_idx in range(matrix.shape[1]):
            row[f"{prefix}_target_{target_idx}"] = float(matrix[draft_idx, target_idx].item())
        rows.append(row)
    return rows


def layer_map_rows(layer_map: Sequence[int], similarity: torch.Tensor, depth_map: Sequence[int]) -> List[Dict[str, Any]]:
    rows = []
    for draft_idx, target_idx in enumerate(layer_map):
        depth_target_idx = int(depth_map[draft_idx])
        rows.append(
            {
                "draft_layer": int(draft_idx),
                "learned_target_layer": int(target_idx),
                "learned_similarity": float(similarity[draft_idx, target_idx].item()),
                "depth_target_layer": depth_target_idx,
                "depth_similarity": float(similarity[draft_idx, depth_target_idx].item()),
            }
        )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Learn monotonic Qwen target-to-draft layer maps for KV Reduce.")
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
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--num_sequences", type=int, default=32)
    parser.add_argument("--position_stride", type=int, default=8)
    parser.add_argument("--max_rows_per_layer", type=int, default=512)
    parser.add_argument("--streaming", "--stream_eval", dest="streaming", action="store_true")
    parser.add_argument(
        "--features",
        type=str,
        default="k_pre,v_pre,attn_out,residual",
        help="Comma-separated features from: k_pre,v_pre,attn_out,residual.",
    )
    parser.add_argument(
        "--feature_weights",
        type=str,
        default=None,
        help='Optional comma-separated weights, e.g. "k_pre=1.2,attn_out=1.0,residual=0.5".',
    )
    parser.add_argument("--allow_layer_reuse", action="store_true")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow_incompatible_tokenizers", action="store_true")
    parser.add_argument("--out_dir", type=str, default="outputs/layer_map")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="kv-reduce")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default="layer-map")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    wandb_run = init_wandb(args)

    features = parse_csv_items(args.features)
    allowed_features = set(DEFAULT_FEATURE_WEIGHTS)
    unknown = sorted(set(features) - allowed_features)
    if unknown:
        raise ValueError(f"Unsupported features: {unknown}. Choices: {sorted(allowed_features)}")
    feature_weights = parse_feature_weights(features, args.feature_weights)

    print("Loading models and tokenizers...")
    big_tokenizer = load_tokenizer(args.big_model)
    small_tokenizer = load_tokenizer(args.small_model)
    compatibility = tokenizer_compatibility_report(big_tokenizer, small_tokenizer)
    if (not compatibility["all_probe_encodings_match"]) and (not args.allow_incompatible_tokenizers):
        raise ValueError("Tokenizers appear incompatible.")

    big_model = load_causal_lm(
        args.big_model,
        device=args.big_device,
        dtype_name=args.big_dtype,
        attn_implementation="eager",
    )
    small_model = load_causal_lm(
        args.small_model,
        device=args.small_device,
        dtype_name=args.small_dtype,
        attn_implementation="eager",
    )

    target_layers = int(big_model.config.num_hidden_layers)
    draft_layers = int(small_model.config.num_hidden_layers)
    target_store: Dict[int, Dict[str, List[torch.Tensor]]] = defaultdict(lambda: defaultdict(list))
    draft_store: Dict[int, Dict[str, List[torch.Tensor]]] = defaultdict(lambda: defaultdict(list))

    hooks = []
    for layer_idx, layer in enumerate(big_model.model.layers):
        hooks.append(
            layer.self_attn.register_forward_hook(
                make_attention_hook(
                    layer_idx=layer_idx,
                    features=features,
                    store=target_store,
                    position_stride=args.position_stride,
                    max_rows=args.max_rows_per_layer,
                ),
                with_kwargs=True,
            )
        )
    for layer_idx, layer in enumerate(small_model.model.layers):
        hooks.append(
            layer.self_attn.register_forward_hook(
                make_attention_hook(
                    layer_idx=layer_idx,
                    features=features,
                    store=draft_store,
                    position_stride=args.position_stride,
                    max_rows=args.max_rows_per_layer,
                ),
                with_kwargs=True,
            )
        )

    print("Collecting pre-RoPE/content representations...")
    blocks = iter_token_blocks(
        tokenizer=big_tokenizer,
        seq_len=args.seq_len,
        max_blocks=args.num_sequences,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.split,
        text_file=args.text_file,
        text_column=args.text_column,
        shuffle=args.shuffle,
        seed=args.seed,
        streaming=args.streaming,
    )
    for step, block in enumerate(blocks, start=1):
        with torch.no_grad():
            _ = big_model(input_ids=block.unsqueeze(0).to(args.big_device), use_cache=False)
            _ = small_model(input_ids=block.unsqueeze(0).to(args.small_device), use_cache=False)
        if step % 10 == 0:
            print(f"  collected {step} / {args.num_sequences} sequences")

    for hook in hooks:
        hook.remove()

    print("Computing CKA similarity matrix...")
    target_features = materialize_features(target_store, num_layers=target_layers, features=features)
    draft_features = materialize_features(draft_store, num_layers=draft_layers, features=features)
    similarity, per_feature = compute_similarity_matrix(
        target_features=target_features,
        draft_features=draft_features,
        feature_weights=feature_weights,
    )

    learned_map = monotonic_layer_map(similarity, allow_layer_reuse=args.allow_layer_reuse)
    depth_map = depth_layer_map(target_layers, draft_layers)
    learned_score = float(sum(similarity[i, j].item() for i, j in enumerate(learned_map)) / len(learned_map))
    depth_score = float(sum(similarity[i, j].item() for i, j in enumerate(depth_map)) / len(depth_map))

    summary = {
        "big_model": args.big_model,
        "small_model": args.small_model,
        "num_target_layers": target_layers,
        "num_draft_layers": draft_layers,
        "layer_map": learned_map,
        "depth_layer_map": depth_map,
        "learned_mean_similarity": learned_score,
        "depth_mean_similarity": depth_score,
        "mean_similarity_gain": learned_score - depth_score,
        "features": features,
        "feature_weights": feature_weights,
        "seq_len": int(args.seq_len),
        "num_sequences": int(args.num_sequences),
        "position_stride": int(args.position_stride),
        "max_rows_per_layer": int(args.max_rows_per_layer),
    }

    write_json(summary, os.path.join(args.out_dir, "layer_map.json"))
    write_csv(matrix_to_rows(similarity), os.path.join(args.out_dir, "similarity_matrix.csv"))
    write_csv(layer_map_rows(learned_map, similarity, depth_map), os.path.join(args.out_dir, "layer_map_rows.csv"))
    for feature, matrix in per_feature.items():
        write_csv(matrix_to_rows(matrix, prefix=feature), os.path.join(args.out_dir, f"{feature}_cka_matrix.csv"))

    if wandb_run is not None:
        wandb_run.summary["learned_mean_similarity"] = learned_score
        wandb_run.summary["depth_mean_similarity"] = depth_score
        wandb_run.summary["mean_similarity_gain"] = learned_score - depth_score
        wandb_run.summary["layer_map"] = learned_map
        try:
            import wandb

            rows = layer_map_rows(learned_map, similarity, depth_map)
            wandb_run.log(
                {
                    "layer_map/rows": wandb.Table(
                        columns=list(rows[0].keys()),
                        data=[list(row.values()) for row in rows],
                    )
                }
            )
        except Exception:
            pass

    print("Done!")
    print(f"  learned map score: {learned_score:.4f}")
    print(f"  depth map score:   {depth_score:.4f}")
    print(f"  {os.path.join(args.out_dir, 'layer_map.json')}")


if __name__ == "__main__":
    main()
