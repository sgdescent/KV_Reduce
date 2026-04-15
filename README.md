# KV Cache Perturbation + Linear Translation Experiments

This mini bundle gives you two runnable experiments for the project idea:

1. `kv_perturbation_sweep.py`
   - Takes a model KV cache, perturbs keys/values by controlled noise, and measures how much the next-token distribution and greedy continuation change.
   - Useful for answering: *"How stable is generation to small KV perturbations?"*

2. `fit_kv_linear_probe.py`
   - Fits a **layerwise linear affine map** from the big model's KV cache to the small model's KV cache using closed-form ridge regression.
   - Evaluates both:
     - cache reconstruction quality
     - functional quality: how the small model behaves when you replace its native prefix KV cache with translated KV from the big model.

3. `fit_kv_factorized_probe.py`
   - Trains a **streamed low-rank neural translator** for the KV cache with a bottleneck of shape `D -> R -> D` per layer (separate K/V modules).
   - Designed for much larger Hugging Face corpora by reading token blocks online instead of materializing the full split in memory.
   - Evaluates both:
     - per-layer cache reconstruction
     - next-token behavior when the small model consumes translated big-model KV

## Recommended starter pair

A very convenient pair is:

- Big: `Qwen/Qwen2.5-3B`
- Small: `Qwen/Qwen2.5-1.5B`

Why this pair is nice:

- same tokenizer family
- same number of KV heads (`2`)
- same head dimension for K/V (`128`)
- so the **per-layer KV tensor shape matches exactly**

That makes it a genuine KV-sharing experiment rather than a shape-matching workaround.

## Install

```bash
pip install -U torch transformers datasets accelerate sentencepiece safetensors
```

## 1) KV perturbation sweep

Single GPU:

```bash
python kv_perturbation_sweep.py \
  --model Qwen/Qwen2.5-1.5B \
  --dataset_name wikitext \
  --dataset_config wikitext-2-raw-v1 \
  --split validation \
  --seq_len 256 \
  --num_sequences 128 \
  --alphas 0,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1 \
  --perturb_targets keys,values,both \
  --layer_mode all \
  --generate_steps 16 \
  --out_dir outputs/perturb_qwen15b
```

Per-layer sensitivity sweep:

```bash
python kv_perturbation_sweep.py \
  --model Qwen/Qwen2.5-1.5B \
  --dataset_name wikitext \
  --dataset_config wikitext-2-raw-v1 \
  --split validation \
  --seq_len 256 \
  --num_sequences 64 \
  --alphas 1e-3,3e-3,1e-2 \
  --perturb_targets both \
  --layer_mode single \
  --generate_steps 8 \
  --out_dir outputs/perturb_qwen15b_single_layers
```

Important output columns:

- `accept_mass`: one-step speculative acceptance proxy, `sum_i min(p_i, q_i)`
- `tv`: total variation distance, equal to `1 - accept_mass`
- `top1_match`: whether the argmax token stayed the same
- `gen_prefix_match_len`: how many greedy-generated tokens remain identical before first divergence

Sanity check: the `alpha=0` rows are a built-in correctness check. Because the baseline
and all perturbed forward passes use the same cache reconstruction path (legacy round-trip)
and the same explicit `cache_position` / `attention_mask`, the alpha=0 row must produce
near-zero divergence (`kl≈0`, `js≈0`, `accept_mass≈1`, `top1_match≈1`,
`gen_exact_match≈1`). If it does not, there is a cache-format or decode-position mismatch
in the environment.

## 2) Fit a linear KV translator

Single GPU:

```bash
python fit_kv_linear_probe.py \
  --big_model Qwen/Qwen2.5-3B \
  --small_model Qwen/Qwen2.5-1.5B \
  --dataset_name wikitext \
  --dataset_config wikitext-2-raw-v1 \
  --train_split train \
  --eval_split validation \
  --seq_len 256 \
  --train_sequences 512 \
  --eval_sequences 128 \
  --lambda_reg 1e-4 \
  --out_dir outputs/qwen25_3b_to_15b_probe
```

Two GPUs (big model on one GPU, small model on another):

```bash
srun --gres=gpu:2 --cpus-per-task=16 --mem=64G \
  python fit_kv_linear_probe.py \
  --big_model Qwen/Qwen2.5-3B \
  --small_model Qwen/Qwen2.5-1.5B \
  --big_device cuda:0 \
  --small_device cuda:1 \
  --dataset_name wikitext \
  --dataset_config wikitext-2-raw-v1 \
  --train_split train \
  --eval_split validation \
  --seq_len 256 \
  --train_sequences 512 \
  --eval_sequences 128 \
  --lambda_reg 1e-4 \
  --out_dir outputs/qwen25_3b_to_15b_probe_2gpu
```

Useful outputs:

- `translator.pt`
  - learned affine translator weights per layer for K and V
- `summary.json`
  - aggregate metrics
- `reconstruction_per_layer.csv`
  - per-layer KV reconstruction metrics
- `next_token_rows.csv`
  - actual behavior when the small model consumes **translated big-model KV**

Most important metrics to watch in `next_token_rows.csv` / `summary.json`:

- `native_vs_big_accept_mass`
  - how aligned the untouched small model is to the big model
- `translated_vs_big_accept_mass`
  - whether translated big KV makes the small model more aligned to the big model
- `translated_vs_native_accept_mass`
  - whether translated big KV approximates the small model's own prefix computation
- `translated_vs_big_top1_match`
  - greedy-token alignment with the big model

## Suggested experiment order

1. Run perturbation on the small model first.
2. Run perturbation on the big model.
3. Fit the linear probe big -> small.
4. Compare:
   - native small vs big
   - identity big-KV -> small (if dims match)
   - learned linear big-KV -> small

## What would count as a promising result?

- `alpha=0` rows are exact identity (sanity check, see above).
- Small perturbations (`alpha ≤ 0.01`) leave `accept_mass` and `top1_match` close to their
  alpha=0 values, with a sharp drop only at larger noise levels.
- Value perturbations are more robust than key perturbations — keys drive attention routing
  so errors there compound faster.
- Learned linear translation beats the identity baseline on `k_cos` / `v_cos`.
- `translated_vs_big_accept_mass` improves over `native_vs_big_accept_mass`.
- Per-layer reconstruction is especially good in early/mid layers.

## Natural next steps after this bundle

- add a logit-space training objective instead of only cache regression
- try low-rank translators instead of full affine maps
- learn layer alignment instead of fixed depth alignment
- test draft acceptance over multi-token speculative blocks instead of one-step proxies

## 3) Train a streamed low-rank KV translator

This trainer is intended for the "more data + fewer parameters" version of the project:

- it supports Hugging Face streaming datasets
- it trains a separate low-rank residual network per layer
- each translator is approximately `D x R + R x D` parameters per target (plus optional bias / layernorm)

Example command:

```bash
python fit_kv_factorized_probe.py \
  --big_model Qwen/Qwen2.5-3B \
  --small_model Qwen/Qwen2.5-1.5B \
  --big_device cuda:0 \
  --small_device cuda:1 \
  --dataset_name wikitext \
  --dataset_config wikitext-2-raw-v1 \
  --train_split train \
  --eval_split validation \
  --stream_train \
  --seq_len 256 \
  --train_sequences 20000 \
  --eval_sequences 128 \
  --shuffle_train \
  --shuffle_buffer_size 10000 \
  --position_stride 2 \
  --max_rows_per_layer_per_block 128 \
  --rank 64 \
  --lr 3e-4 \
  --weight_decay 1e-4 \
  --cos_loss_weight 0.1 \
  --out_dir outputs/qwen25_3b_to_15b_factorized_stream
```

For a much larger corpus, replace `--dataset_name` / `--dataset_config` with your preferred Hugging Face dataset and keep `--stream_train` enabled.

Most useful outputs:

- `factorized_translator.pt`
  - learned low-rank per-layer translator weights
- `parameter_summary.csv`
  - parameter counts per layer for K, V, and total
- `train_log.csv`
  - rolling training metrics
- `train_per_layer.csv`
  - average train losses / cosine scores per layer
- `reconstruction_per_layer.csv`
  - per-layer reconstruction metrics on the eval split
- `next_token_rows.csv`
  - functional behavior when the small model consumes translated big-model KV
- `summary.json`
  - aggregate metrics and run metadata
