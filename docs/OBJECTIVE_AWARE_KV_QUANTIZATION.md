# Objective-Aware KV-Cache Quantization

## Research question

KV-cache quantization is usually optimized for ordinary language-model quality,
such as perplexity or next-token KL. Speculative decoding uses a different
downstream objective: the draft distribution should maximize acceptance under
the target verifier. This pipeline tests whether those objectives induce
different precision allocations on the same draft model.

For every candidate `(layer, component, bits)`, we measure:

```text
quality risk     = KL(logits_BF16 || logits_quantized)
speculative risk = acceptance_BF16 - acceptance_quantized
```

Held-out delta NLL is still the primary ordinary-quality outcome in the final
cross-evaluation. KL is used for sensitivity ranking because it is nonnegative
and substantially less noisy than a small calibration sample's delta NLL.

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

## Run the robustness matrix

After one profiling campaign finishes, reuse its sensitivity maps across equal
memory budgets, held-out seeds, and context lengths:

```bash
SOURCE_ROOT=outputs/objective_kv/qwen25_objective_1k_v1 \
MATRIX_ROOT=outputs/objective_kv/qwen25_matrix_v1 \
BUDGETS=6,8,10,12 \
CONTEXTS=512,1024,4096 \
SEEDS=0,1,2 \
NUM_EVAL=32 \
MAX_CONCURRENT=2 \
bash scripts/submit_objective_kv_matrix.sh
```

This creates 72 GPU evaluation tasks but allows only two to run concurrently.
The final aggregation reports seed confidence intervals and paired objective
advantages at each memory budget and context length.
