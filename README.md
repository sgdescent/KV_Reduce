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

- Small perturbations leave `accept_mass` close to `1.0` and `top1_match` high.
- Learned linear translation beats the identity baseline.
- `translated_vs_big_accept_mass` improves over `native_vs_big_accept_mass`.
- Per-layer reconstruction is especially good in early/mid layers.

## Natural next steps after this bundle

- add a logit-space training objective instead of only cache regression
- try low-rank translators instead of full affine maps
- learn layer alignment instead of fixed depth alignment
- test draft acceptance over multi-token speculative blocks instead of one-step proxies
