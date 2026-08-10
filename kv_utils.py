import json
import math
import os
import random
import warnings
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    try:
        from transformers import DynamicCache
    except ImportError:
        DynamicCache = None
except ImportError as e:
    raise ImportError(
        "transformers is required for these experiments. Install it with: pip install transformers accelerate sentencepiece safetensors"
    ) from e


DTYPE_MAP = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def resolve_dtype(dtype_name: str) -> torch.dtype:
    key = dtype_name.lower()
    if key not in DTYPE_MAP:
        raise ValueError(f"Unsupported dtype: {dtype_name}. Choices: {sorted(DTYPE_MAP)}")
    return DTYPE_MAP[key]



def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer



def load_causal_lm(
    model_name: str,
    device: str,
    dtype_name: str = "bf16",
    attn_implementation: Optional[str] = None,
):
    dtype = resolve_dtype(dtype_name)
    kwargs = dict(torch_dtype=dtype, low_cpu_mem_usage=True)
    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.to(device)
    return model



def freeze_model(model) -> None:
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)



def get_num_kv_heads(config) -> int:
    return int(getattr(config, "num_key_value_heads", config.num_attention_heads))



def get_head_dim(config) -> int:
    return int(config.hidden_size // config.num_attention_heads)



def get_kv_dim(config) -> int:
    return get_num_kv_heads(config) * get_head_dim(config)



def depth_layer_map(num_big_layers: int, num_small_layers: int) -> List[int]:
    if num_small_layers == 1:
        return [num_big_layers - 1]
    mapped = []
    for s in range(num_small_layers):
        b = round(s * (num_big_layers - 1) / (num_small_layers - 1))
        mapped.append(int(b))
    return mapped



def as_legacy_cache(past_key_values):
    """Return past_key_values as a plain tuple-of-tuples ((k, v), ...) regardless of transformers version."""
    if past_key_values is None:
        return None
    # transformers 5.x: DynamicCache has .layers with .keys/.values per layer
    if hasattr(past_key_values, "layers"):
        return tuple((layer.keys, layer.values) for layer in past_key_values.layers)
    # transformers 4.36-4.x: DynamicCache with to_legacy_cache()
    if hasattr(past_key_values, "to_legacy_cache"):
        return past_key_values.to_legacy_cache()
    # transformers < 4.36: already a plain tuple
    return past_key_values


def legacy_to_cache(legacy_cache):
    """Convert a legacy tuple-of-tuples ((k, v), ...) back to whatever cache object the model expects."""
    if DynamicCache is None:
        # transformers < 4.36: model accepts a plain tuple directly
        return tuple((layer_cache[0], layer_cache[1]) for layer_cache in legacy_cache)
    # transformers 5.x: use ddp_cache_data constructor
    if hasattr(DynamicCache, "layers") or not hasattr(DynamicCache, "from_legacy_cache"):
        return DynamicCache(ddp_cache_data=legacy_cache)
    # transformers 4.36-4.x
    return DynamicCache.from_legacy_cache(legacy_cache)



def clone_legacy_cache(legacy_cache) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
    return tuple((layer_cache[0].clone(), layer_cache[1].clone()) for layer_cache in legacy_cache)



def flatten_kv(kv: torch.Tensor) -> torch.Tensor:
    # [B, H, T, D] -> [B*T, H*D]
    bsz, num_heads, seqlen, head_dim = kv.shape
    return kv.permute(0, 2, 1, 3).reshape(bsz * seqlen, num_heads * head_dim)



def unflatten_kv(flat: torch.Tensor, bsz: int, seqlen: int, num_heads: int, head_dim: int) -> torch.Tensor:
    # [B*T, H*D] -> [B, H, T, D]
    return flat.reshape(bsz, seqlen, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()



def topk_overlap(logits_a: torch.Tensor, logits_b: torch.Tensor, k: int = 5) -> torch.Tensor:
    topk_a = torch.topk(logits_a, k=k, dim=-1).indices
    topk_b = torch.topk(logits_b, k=k, dim=-1).indices
    overlaps = []
    for a_row, b_row in zip(topk_a, topk_b):
        a_set = set(a_row.tolist())
        b_set = set(b_row.tolist())
        overlaps.append(len(a_set.intersection(b_set)) / float(k))
    return torch.tensor(overlaps, device=logits_a.device, dtype=torch.float32)



def distribution_metrics(target_logits: torch.Tensor, draft_logits: torch.Tensor, topk: int = 5) -> Dict[str, float]:
    # Interprets target_logits as p and draft_logits as q.
    p_log = F.log_softmax(target_logits.float(), dim=-1)
    q_log = F.log_softmax(draft_logits.float(), dim=-1)
    p = p_log.exp()
    q = q_log.exp()
    m = 0.5 * (p + q)
    m_log = torch.log(m.clamp_min(1e-12))

    kl_pq = F.kl_div(q_log, p, reduction="none").sum(dim=-1)
    kl_qp = F.kl_div(p_log, q, reduction="none").sum(dim=-1)
    js = 0.5 * (
        F.kl_div(m_log, p, reduction="none").sum(dim=-1)
        + F.kl_div(m_log, q, reduction="none").sum(dim=-1)
    )
    tv = 0.5 * (p - q).abs().sum(dim=-1)
    accept_mass = torch.minimum(p, q).sum(dim=-1)
    top1_match = (target_logits.argmax(dim=-1) == draft_logits.argmax(dim=-1)).float()
    topk_match = topk_overlap(target_logits, draft_logits, k=topk)

    target_top1 = target_logits.argmax(dim=-1, keepdim=True)
    draft_prob_on_target_top1 = q.gather(-1, target_top1).squeeze(-1)
    target_prob_on_target_top1 = p.gather(-1, target_top1).squeeze(-1)

    return {
        "kl_p_to_q": float(kl_pq.mean().item()),
        "kl_q_to_p": float(kl_qp.mean().item()),
        "js": float(js.mean().item()),
        "tv": float(tv.mean().item()),
        "accept_mass": float(accept_mass.mean().item()),
        "top1_match": float(top1_match.mean().item()),
        f"top{topk}_overlap": float(topk_match.mean().item()),
        "draft_prob_on_target_top1": float(draft_prob_on_target_top1.mean().item()),
        "target_prob_on_target_top1": float(target_prob_on_target_top1.mean().item()),
    }



def first_mismatch_position(seq_a: Sequence[int], seq_b: Sequence[int]) -> int:
    for idx, (a, b) in enumerate(zip(seq_a, seq_b)):
        if a != b:
            return idx
    return min(len(seq_a), len(seq_b))



def write_json(data, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)



def write_jsonl(rows: Iterable[Dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")



def find_text_column(example: Dict) -> str:
    if "text" in example and isinstance(example["text"], str):
        return "text"
    for key, value in example.items():
        if isinstance(value, str):
            return key
    raise ValueError("Could not infer a text column. Please use a dataset with a string column or a text file.")



def iter_token_blocks(
    tokenizer,
    seq_len: int,
    max_blocks: int,
    dataset_name: Optional[str] = None,
    dataset_config: Optional[str] = None,
    split: str = "train",
    text_file: Optional[str] = None,
    text_column: Optional[str] = None,
    add_eos_between_examples: bool = True,
    shuffle: bool = False,
    seed: int = 0,
    streaming: bool = False,
    shuffle_buffer_size: int = 10_000,
    split_fallbacks: Optional[Sequence[str]] = None,
    skip_blocks: int = 0,
) -> Iterator[torch.Tensor]:
    if text_file is None and dataset_name is None:
        raise ValueError("Provide either --text_file or --dataset_name.")
    if skip_blocks < 0:
        raise ValueError("skip_blocks must be >= 0")

    if isinstance(dataset_config, str) and dataset_config.strip().lower() in {"", "none", "null"}:
        dataset_config = None

    if text_file is not None:
        with open(text_file, "r", encoding="utf-8") as f:
            texts = f.readlines()
    else:
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise ImportError("The datasets package is required when using --dataset_name. Install it with: pip install datasets") from e
        split_candidates = [split]
        for fallback in split_fallbacks or []:
            if fallback and fallback not in split_candidates:
                split_candidates.append(fallback)

        dataset = None
        used_split = split
        last_error = None
        for candidate_split in split_candidates:
            try:
                dataset = load_dataset(dataset_name, dataset_config, split=candidate_split, streaming=streaming)
                used_split = candidate_split
                break
            except ValueError as e:
                last_error = e
                if "Unknown split" not in str(e):
                    raise

        if dataset is None:
            raise last_error

        if used_split != split:
            warnings.warn(
                f'Dataset split "{split}" was unavailable for {dataset_name}; using "{used_split}" instead.',
                stacklevel=2,
            )

        if streaming:
            if shuffle:
                dataset = dataset.shuffle(buffer_size=shuffle_buffer_size, seed=seed)
            iterator = iter(dataset)
            first = next(iterator, None)
            if first is None:
                raise ValueError("Dataset split is empty.")
            column = text_column or find_text_column(first)

            def iter_texts():
                try:
                    yield first.get(column)
                    for example in iterator:
                        yield example.get(column)
                finally:
                    close = getattr(iterator, "close", None)
                    if close is not None:
                        close()

            texts = iter_texts()
        else:
            if shuffle:
                dataset = dataset.shuffle(seed=seed)
            if len(dataset) == 0:
                raise ValueError("Dataset split is empty.")
            first = dataset[0]
            column = text_column or find_text_column(first)
            texts = dataset[column]

    buffer: List[int] = []
    yielded = 0
    seen_blocks = 0
    eos_id = tokenizer.eos_token_id

    try:
        for text in texts:
            if text is None:
                continue
            text = str(text)
            if not text.strip():
                continue
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            if not ids:
                continue
            buffer.extend(ids)
            if add_eos_between_examples and eos_id is not None:
                buffer.append(eos_id)
            while len(buffer) >= seq_len:
                block = torch.tensor(buffer[:seq_len], dtype=torch.long)
                if seen_blocks >= skip_blocks:
                    yield block
                    yielded += 1
                    if yielded >= max_blocks:
                        return
                seen_blocks += 1
                buffer = buffer[seq_len:]
    finally:
        close = getattr(texts, "close", None)
        if close is not None:
            close()



def tokenizer_compatibility_report(tok_a, tok_b) -> Dict[str, object]:
    probes = [
        "The quick brown fox jumps over the lazy dog.",
        "Speculative decoding with shared KV cache.",
        "Hello world! 12345",
    ]
    matches = []
    for probe in probes:
        ids_a = tok_a(probe, add_special_tokens=False)["input_ids"]
        ids_b = tok_b(probe, add_special_tokens=False)["input_ids"]
        matches.append(ids_a == ids_b)
    return {
        "vocab_size_a": int(tok_a.vocab_size),
        "vocab_size_b": int(tok_b.vocab_size),
        "same_vocab_size": int(tok_a.vocab_size) == int(tok_b.vocab_size),
        "all_probe_encodings_match": all(matches),
        "probe_matches": matches,
    }


@dataclass
class RidgeAccumulator:
    d_in: int
    d_out: int
    lambda_reg: float = 1e-4

    def __post_init__(self):
        self.A = torch.zeros(self.d_in + 1, self.d_in + 1, dtype=torch.float64)
        self.B = torch.zeros(self.d_in + 1, self.d_out, dtype=torch.float64)
        self.num_rows = 0

    def update(self, x: torch.Tensor, y: torch.Tensor) -> None:
        if x.ndim != 2 or y.ndim != 2:
            raise ValueError("x and y must both be rank-2 tensors.")
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must have the same number of rows.")
        x_cpu = x.detach().to(dtype=torch.float64, device="cpu")
        y_cpu = y.detach().to(dtype=torch.float64, device="cpu")
        ones = torch.ones(x_cpu.shape[0], 1, dtype=torch.float64)
        x_aug = torch.cat([x_cpu, ones], dim=1)
        self.A += x_aug.T @ x_aug
        self.B += x_aug.T @ y_cpu
        self.num_rows += int(x.shape[0])

    def solve(self) -> Tuple[torch.Tensor, torch.Tensor]:
        reg = torch.eye(self.d_in + 1, dtype=torch.float64) * self.lambda_reg
        reg[-1, -1] = 0.0  # do not regularize bias
        w_aug = torch.linalg.solve(self.A + reg, self.B)  # [d_in+1, d_out]
        weight = w_aug[:-1].T.to(dtype=torch.float32).contiguous()  # [d_out, d_in]
        bias = w_aug[-1].to(dtype=torch.float32).contiguous()  # [d_out]
        return weight, bias



def affine_apply(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return x @ weight.T + bias



def safe_mean(values: List[float]) -> float:
    if not values:
        return float("nan")
    return float(sum(values) / len(values))
