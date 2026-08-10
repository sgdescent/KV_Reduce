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

## Joint target/draft role allocation

Draft-only quantization cannot change the intended target distribution; it only
changes proposal quality and acceptance. Quantizing the target cache can save a
much larger fraction of total KV memory, but may also move the verifier logits.
The joint experiment therefore evaluates every target/draft pair directly:

```text
maximize total target + draft KV memory saved
subject to target KL / token-fidelity budget
           and speculative-acceptance budget
```

`benchmark_spec_kv_quantization.py --target_quant_configs ...` crosses target
configs with the existing draft `--quant_configs`. The BF16 target sequence is
generated once outside timing. Each candidate reports acceptance, sequence and
token agreement with that reference, non-tie divergence counts, and separate
target/draft/total cache bytes. Accepted proposal KV is quantized before the
target bonus step; rejected speculative state is cropped before promotion, so
the fake-quantized runtime follows commit-only cache semantics.

The default `--target_quant_configs none` retains the audited
`cached_dynamic_v4` draft-only artifact format. Joint artifacts use
`cached_dynamic_v5_joint_target_draft`; target-quantized output changes are
quality outcomes, while only a non-tie mismatch in the fully BF16 baseline
invalidates a prompt.

Run the three-seed 1K/4K joint grid with:

```bash
ROOT=outputs/kivi_joint_target_draft/qwen25_3b_15b \
ARRAY_THROTTLE=1 \
bash scripts/submit_joint_target_draft_grid.sh
```

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

For a paper result, use a streaming held-out source and set each skip base past
the blocks consumed by calibration. The manifest then assigns non-overlapping
contiguous shards to every seed, including speculative warmup prompts. Explicit
shards deliberately disable dataset shuffling so offsets remain disjoint:

```bash
DATASET_NAME=HuggingFaceFW/fineweb-edu \
DATASET_CONFIG=sample-10BT \
EVAL_SPLIT=train \
STREAM_EVAL=1 \
QUALITY_SKIP_BASE=1024 \
ACCEPTANCE_SKIP_BASE=2048 \
bash scripts/submit_objective_kv_matrix.sh
```

The skip bases are experiment metadata, not universal defaults; choose values
larger than the corresponding calibration consumption. Older 9- and 11-column
manifests remain runnable, but they are diagnostic because seed shuffling does
not prove example-level disjointness on a finite split.
