# Latent Prefix KV Sharing Revamp

## Thesis

Speculative decoding duplicates the long prefix KV cache: one copy for the target model and one
copy for the draft model. KV Reduce should target that duplication directly.

The revised method is **Latent Prefix KV Sharing for Speculative Decoding**:

- The target model owns the accepted prefix KV cache.
- Shared draft layers read a learned view of the target prefix cache.
- Draft-only speculative tokens use a tiny temporary tail cache of length at most `draft_steps`.
- After verification, accepted tokens are committed to the target cache and rejected draft-tail state
  is discarded.

This keeps final generation exact because target verification is unchanged. The learned shared-cache
reader only affects acceptance rate and speed.

## Runtime Model

Native speculative decoding stores:

```text
target full-prefix KV + draft full-prefix KV
```

KV Reduce stores:

```text
target full-prefix KV + draft unshared-layer KV + shared-layer draft tail KV
```

For shared layers, attention reads:

```text
[target prefix memory through learned reader] + [temporary native draft tail memory]
```

The new cached simulator is exposed through:

```bash
python eval_absorbed_spec_decode.py \
  --translator_path outputs/kv_absorbed/absorbed_translator.pt \
  --shared_layers top:4 \
  --shared_variant full \
  --norm_match rms \
  --absorbed_cache_mode prefix \
  --wandb --wandb_project kv-reduce
```

`--absorbed_cache_mode recompute` keeps the old prototype path as an ablation.

## Layer Mapping

The old depth map assumes draft layer `i` corresponds to a linearly spaced target layer. The revamp
adds a learned monotonic map using CKA over content-side representations:

- `k_pre`: key projection before RoPE, so positional rotation is not mixed into the score.
- `v_pre`: value projection/content payload.
- `attn_out`: final attention module output.
- `residual`: attention input residual stream.

Run:

```bash
python learn_layer_map.py \
  --big_model Qwen/Qwen2.5-3B \
  --small_model Qwen/Qwen2.5-1.5B \
  --dataset_name HuggingFaceFW/fineweb-edu \
  --dataset_config sample-10BT \
  --num_sequences 128 \
  --seq_len 512 \
  --streaming \
  --out_dir outputs/layer_map_qwen25_3b_15b \
  --wandb --wandb_project kv-reduce
```

Then train the absorbed translator with:

```bash
python fit_kv_absorbed.py \
  --big_model Qwen/Qwen2.5-3B \
  --small_model Qwen/Qwen2.5-1.5B \
  --dataset_name HuggingFaceFW/fineweb-edu \
  --dataset_config sample-10BT \
  --train_sequences 50000 \
  --seq_len 512 \
  --stream_train \
  --output_routing_source shared \
  --layer_map_file outputs/layer_map_qwen25_3b_15b/layer_map.json \
  --out_dir outputs/kv_absorbed_learned_map \
  --wandb --wandb_project kv-reduce
```

## Cached Benchmark

Use the benchmark script for apples-to-apples runtime and memory logging:

```bash
python benchmark_absorbed_spec_decode.py \
  --translator_path outputs/kv_absorbed_learned_map/absorbed_translator.pt \
  --dataset_name wikitext \
  --dataset_config wikitext-2-raw-v1 \
  --prompt_len 1024 \
  --num_prompts 200 \
  --shared_layers top:4 \
  --shared_variant full \
  --norm_match rms \
  --absorbed_cache_mode prefix \
  --out_dir outputs/benchmark_prefix_top4_rms \
  --wandb --wandb_project kv-reduce \
  --wandb_group cached-prefix-benchmark
```

The benchmark reports:

- `target_calls`, `draft_calls`, and `target_recompute_avoided`
- latency and tokens/sec
- acceptance rate, accepted tokens per round, top-1 match, JS/KL drift, accept mass
- peak CUDA memory
- analytical KV-cache bytes for native SpecDec vs KV Reduce

## Long-Context Projection

Analytical memory projections can be generated without loading weights:

```bash
python estimate_long_context_kv_savings.py \
  --big_model Qwen/Qwen2.5-7B \
  --small_model Qwen/Qwen2.5-3B \
  --contexts 512,1024,4096,8192,16384,32768 \
  --shared_layers_specs top:4,top:8,all \
  --draft_tail_len 4 \
  --out_dir outputs/long_context_qwen25_7b_3b
```

This produces CSV/JSON plus a poster-ready plot when `matplotlib` is installed.

## Next Research Step

The current implementation still uses closed-form absorbed K/O adapters. The next serious adapter
should initialize from this solution and train a small frozen-model module with:

```text
attention-output MSE
attention-logit KL
native-draft next-token KL
target next-token KL
residual RMS/std matching
acceptance-mass proxy
```

That is the path toward the MLA-inspired latent prefix cache rather than the current full-value
reader.
