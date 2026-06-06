# KV Reduce High-Savings Revamp

## Why The Current Savings Look Small

The current cached-prefix implementation is exact with respect to the original target model:

```text
native SpecDec: target full KV + draft full KV
KV Reduce:      target full KV + tiny/partial draft KV
```

That is the safest correctness story, because target verification is unchanged. But it also creates
a hard ceiling: total KV-cache savings cannot exceed the fraction of memory used by the draft cache.
For Qwen target/draft pairs, this ceiling is often only about 35-45% even if the entire draft prefix
cache disappears.

This means the project needs two separate claims:

- **Exact KV Reduce:** preserves the original target model and removes duplicate draft-prefix KV.
- **Aggressive KV Reduce:** compresses the target prefix cache too, using MLA-style or quantized KV.
  This can produce much larger savings, but the verified model is now a compressed target unless the
  compression is lossless.

Keeping these separate makes the paper honest and stronger.

## New Thesis

Speculative decoding has two memory problems:

1. It duplicates the prefix cache across target and draft models.
2. At long context, the target prefix cache dominates serving memory and HBM traffic.

The revised research direction is:

```text
Shared latent prefix memory for speculative decoding.
```

The draft should read a shared prefix representation instead of storing a full draft prefix cache.
For major savings, the target verifier should also use a compressed target prefix cache through a
validated MLA/quantized-cache path.

## Track A: Exact Original Target

This is the near-term systems-safe track.

Runtime:

```text
target full KV + draft tail KV only
```

Properties:

- Final output distribution stays exact because the target cache is unchanged.
- Maximum savings are bounded by the draft-cache fraction.
- The research challenge is acceptance: sharing more draft layers currently causes residual drift.

What we should improve:

- Train the output/value reader harder than the key reader.
- Keep K training small because key alignment saturates early.
- Train O/reader adapters with next-token KL and attention-output losses, not just closed-form MSE.
- Add per-layer gates so each layer can choose native draft KV vs shared prefix KV.
- Use the learned CKA map, but also search non-contiguous layer subsets instead of only `top:k`.

## Track B: Aggressive Compressed Target

This is the high-savings paper track.

Runtime:

```text
compressed target prefix KV + draft tail KV only
```

Options:

- **Target KV quantization:** use int8/int4/int3 cache for the target verifier and the draft reader.
- **Target MLA conversion:** convert Qwen attention into a latent content cache plus partial-RoPE
  positional cache.
- **Hybrid:** MLA latent target cache plus quantized latent cache.

Properties:

- Can plausibly reach 70-90% total KV-cache reduction vs native speculative decoding.
- Not exact with respect to the original target unless compression is lossless.
- Must evaluate target quality separately on LongBench, passkey/needle retrieval, WikiText/C4, and
  task prompts.

Research grounding:

- DeepSeek-V2 reports that MLA reduces KV-cache size dramatically while improving throughput.
- MHA2MLA-style conversion suggests partial-RoPE and SVD/low-rank initialization are the right
  starting point for converting existing attention layers.
- QuantSpec shows that long-context speculative decoding benefits from hierarchical KV
  quantization and keeps high acceptance by using a quantized self-draft path.
- LayerSkip is an important baseline because it avoids the two-model duplicate-cache problem by
  reusing one model's cache.
- EAGLE is an important baseline because it drafts at the feature level rather than trying to make a
  small model maintain a fully separate cache.

## New Experiments

### 1. Ceiling Analysis

Run:

```bash
python estimate_high_savings_design_space.py \
  --big_model Qwen/Qwen2.5-7B \
  --small_model Qwen/Qwen2.5-3B \
  --contexts 1024,4096,8192,16384,32768,65536 \
  --draft_tail_len 4 \
  --target_quant_bits 8,4,3 \
  --target_mla_latent_dims 128,256,512 \
  --target_mla_rope_dim 64 \
  --out_dir outputs/high_savings_qwen25_7b_3b
```

This produces:

```text
high_savings_design_space.csv
high_savings_design_space.json
high_savings_design_space.png
high_savings_vs_context.png
```

### 2. Exact KV Reduce Search

Run quick adapter sweeps:

```text
K sequences: 256
O sequences: 2k, 5k, 20k
shared sets: top:2, top:4, top:6, top:8, learned-greedy
losses: O MSE, next-token KL, residual RMS/std matching
```

Success criterion:

```text
top:8 acceptance within 5-10 percentage points of native draft
```

### 3. Compressed Target Validation

Evaluate compressed target cache independently before combining it with KV Reduce:

```text
target bf16 KV vs target int8/int4 KV
target bf16 KV vs target MLA latent KV
```

Metrics:

```text
next-token top-1 match
JS/KL divergence
passkey/needle retrieval accuracy
LongBench task score
end-to-end target output agreement
```

Quick target-cache quantization diagnostic:

```bash
python eval_target_cache_quantization.py \
  --model Qwen/Qwen2.5-3B \
  --dataset_name wikitext \
  --dataset_config wikitext-2-raw-v1 \
  --num_sequences 128 \
  --prompt_len 1024 \
  --bits 8,4,3 \
  --out_dir outputs/target_cache_quant_qwen25_3b
```

Slurm:

```bash
export BITS="8,4,3"
sbatch --export=ALL,MODEL=Qwen/Qwen2.5-3B,NUM_SEQUENCES=128,PROMPT_LEN=1024,OUT_DIR=outputs/target_cache_quant_qwen25_3b scripts/eval_target_cache_quantization.slurm
```

Only after this passes should we combine with the draft shared-cache reader.

## Stronger Paper Story

The current project should not claim huge savings from top:4 sharing. Instead:

- Top:4 sharing is preliminary evidence that a draft can read target prefix state.
- The publishable method is a two-stage memory hierarchy:

```text
1. Remove duplicate draft prefix KV.
2. Compress the shared target prefix KV.
```

This makes the memory story scale with context length and server batch size, where KV cache is the
real bottleneck.

## Recommended Next Coding Milestones

1. Add a trainable low-rank O/reader adapter initialized from the current closed-form solution.
2. Add a layer-subset search script that selects shared layers by validation acceptance, not by
   `top:k`.
3. Add target-cache quantization eval for int8/int4 with exact diagnostics.
4. Add a minimal Qwen MLA-conversion prototype:
   - split content vs RoPE dimensions,
   - SVD initialize latent down/up projections,
   - train only latent adapters and RMS scales,
   - compare cache bytes and target quality.
5. Make final benchmarks long-context first: 8k, 16k, 32k, 64k.
