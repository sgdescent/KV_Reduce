# Objective-Aware KV-Cache Quantization

## Research question

KV-cache quantization is usually optimized for ordinary language-model quality,
such as perplexity or next-token KL. Speculative decoding uses a different
downstream objective: the draft distribution should maximize acceptance under
the target verifier. This pipeline tests whether those objectives induce
different precision allocations on the same draft model.

For every candidate `(layer, component, bits)`, we measure:

```text
quality risk     = NLL_quantized - NLL_BF16
speculative risk = acceptance_BF16 - acceptance_quantized
```

The quality profile uses teacher-forced continuations with a cache-resident
decode loop. The speculative profile uses the audited cached target/draft loop
in `benchmark_spec_kv_quantization.py`.

## Decisive experiment

1. Profile the same draft model under both objectives.
2. Build a quality-optimized allocation and an acceptance-optimized allocation.
3. Evaluate both allocations under both objectives at the same profiled-component
   mean-bit budget.
4. Report sensitivity-rank correlation, sensitive-component overlap, allocation
   disagreement, NLL/KL, acceptance rate, and packed KV-memory estimates.

The thesis is supported if the acceptance allocation retains materially more
speculative acceptance at equal memory while the quality allocation retains
better NLL/KL. Similar sensitivity maps and cross-evaluation results would
falsify the stronger objective-specific claim.

## Run the development campaign

```bash
bash scripts/submit_objective_kv_campaign.sh
```

Useful overrides:

```bash
TAG=qwen_objective_4k \
PROMPT_LEN=4096 \
NUM_PROFILE=32 \
NUM_EVAL=64 \
LAYERS=top:8 \
WANDB_PROJECT=kv-reduce \
bash scripts/submit_objective_kv_campaign.sh
```

The campaign submits two independent profiling jobs, then dependency-gated
allocation, comparison, and cross-evaluation jobs. By default it excludes known
unreliable Catalyst nodes and uses at most two GPUs concurrently.
