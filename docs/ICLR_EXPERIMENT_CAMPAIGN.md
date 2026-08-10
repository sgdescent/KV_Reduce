# ICLR Experiment Campaign

This campaign tests how quantizer geometry and the downstream decoding objective
change draft-cache key/value precision requirements across model families,
model-size gaps, contexts, and held-out domains.

## Primary hypothesis

The controlled hypothesis is that ordinary LM quality and speculative acceptance
may rank equal-memory K/V allocations differently. Keys determine attention
routing through `softmax(QK^T)`, while values carry the payload after routing,
but bit width alone is not meaningful without fixing axis, grouping, zero-point,
and residual-window choices. Main matched runs therefore use grouped per-channel
affine keys and per-token affine values. The target verifier remains full
precision, so the correction rule is distribution preserving in exact arithmetic;
separate audits measure BF16 differences between batched and tokenwise kernel
paths.

## Model pairs

| ID | Target | Draft |
|---:|---|---|
| 0 | Qwen2.5-3B | Qwen2.5-1.5B |
| 1 | Qwen2.5-7B | Qwen2.5-3B |
| 2 | Qwen3-8B | Qwen3-4B |
| 3 | Llama-3.1-8B | Llama-3.2-3B |
| 4 | OLMo-2-7B | OLMo-2-1B |
| 5 | SmolLM2-1.7B | SmolLM2-360M |

Gemma-2 is omitted because the current Hugging Face account cannot access the
gated 9B checkpoint.

## Stages

1. `smoke`: 256-token context, two prompts, `none/K8V4/K4V8`.
2. `main`: 1K and 4K contexts, all uniform asymmetric baselines.
3. `sensitivity`: one-at-a-time K/V perturbations at 4 and 8 bits on the top
   eight draft layers.
4. `allocation`: acceptance-budgeted mixed-precision search followed by a held
   benchmark against full-precision draft KV.
5. `robustness`: three shuffled seeds for Qwen2.5-3B/1.5B on C4 validation,
   GSM8K questions, and HumanEval prompts. Every domain uses the same KIVI
   geometry under both speculative acceptance and teacher-forced LM quality.
6. `long_context`: exploratory 16K-prefix and 32K-total-sequence Qwen2.5 runs
   on PG19 before increasing the prompt count for the final long-context result.
   The 32K arm uses a 32,752-token prefix plus 16 generated/continuation tokens
   so it remains inside the draft model's declared 32,768-token window.
7. `matched_objectives`: the same 21 uniform K/V configurations are evaluated
   under cached speculative decoding and teacher-forced LM quality at 1K and 4K.
8. `zero_residual`: repeat the equal-memory objective comparisons with no BF16
   key tail. This removes the small byte mismatch introduced by the standard
   128-token KIVI residual window.
9. `standalone_target_quality`: run the same affine grid on Qwen2.5-3B without
   speculative decoding, measuring next-token KL, NLL, and top-1 preservation.
10. `quantizer_factorial`: complete the key-axis/value-zero-point factorial by
    pairing per-token symmetric keys with per-token affine values. Combined with
    the other campaigns, this separates key-axis, value-scheme, and interaction
    effects.
11. `factorial_quality_completion`: profile per-channel affine keys with symmetric
    values under ordinary LM quality, completing all four geometry cells for both
    downstream objectives.
12. `group_residual_sweep`: test key groups `16/32/64/128` with BF16 key-tail
    lengths `0/128` at 1K across three seeds, under both downstream objectives.
13. `cross_family_target_quality`: run the matched affine grid on all six target
    models as ordinary autoregressive LMs. This provides the target-fidelity
    constraints needed for role-aware target/draft precision allocation rather
    than inferring target behavior from the smaller draft checkpoints.
14. `matched_kivi_gamma`: repeat the current per-channel-key/affine-value grid
    at speculative proposal lengths `2/4/8`, with three seeds each. This tests
    whether a precision policy remains stable as quantization errors affect a
    longer unverified draft trajectory.
15. `qwen7_reversal_powered`: replicate the seed-20 Qwen2.5-7B/3B K4V3-versus-
    K3V4 objective reversal with 512 speculative prompts and 256 ordinary-
    quality sequences per seed across three new seeds. This is a predeclared
    confirmation experiment for a small acceptance effect, not an exploratory
    source of additional allocation comparisons.
16. `verifier_exactness_powered`: quantify finite-precision verifier drift on
    32 held-out prompts for each BF16/FP32 and SDPA/eager combination. This
    separates implementation correctness from backend-dependent numerical paths.
17. `multiple_choice_accuracy`: evaluate the standalone Qwen2.5-1.5B cache on
    disjoint HellaSwag and ARC-Challenge examples across three seeds. Each
    example uses a fixed eight-shot training prefix so grouped key quantization
    extends beyond the 128-token BF16 residual window. The evaluator reports raw
    and length-normalized accuracy, paired BF16 agreement, and equal-memory
    K8V4-versus-K4V8 confidence intervals.

The launcher serializes complete stages and caps each stage at two GPUs. Each
stage checks its pair-specific prerequisite artifact; a failed pair is skipped in
later stages without preventing successful pairs from advancing.
Jobs exclude Catalyst nodes `catalyst-0-9` (flaky GPU/prolog behavior observed)
and `catalyst-0-15` (down at campaign launch).

Pairs 0, 4, and 5 fit target and draft on one 24-GiB GPU. Pairs 1, 2, and 3
use two GPUs per job, placing target on `cuda:0` and draft on `cuda:1`. The
two-GPU chain is capped at one concurrent job, so it consumes at most two GPUs.

## Metrics

- speculative token acceptance rate and accepted tokens per verification round;
- target/draft top-1 match, Jensen-Shannon divergence, and acceptance mass;
- HellaSwag and ARC-Challenge multiple-choice accuracy under ordinary decoding;
- draft and total KV-cache bytes saved;
- exact-match against greedy target decoding;
- measured runtime only as a diagnostic, because the current fake-quantization
  path dequantizes in Python and is not a kernel-level speed benchmark.

The evaluator performs one target and one draft prefill per prompt, reuses both
dynamic caches across speculative rounds, verifies each proposal against the
existing target cache, and crops rejected suffixes in place. Target-greedy
reference generation is performed once outside each timed configuration. Runs
missing `runtime.evaluator_version=cached_dynamic_v4` are rejected by the paper
aggregation script.

Grouped keys enter the quantized prefix only after their tokens are committed.
Unverified draft tokens remain in the BF16 residual tail, which can temporarily
grow by at most the proposal length. This prevents a rejected speculative suffix
from irreversibly quantizing a group that should return to the residual window.

Greedy exact-match is reported together with the target top-1 logit margin at
the first mismatch. BF16/SDPA can select a different token when the top logits
are tied; the aggregate distinguishes these numerical ties from non-tie
verification errors instead of silently treating them as algorithm failures.

## Launch

```bash
bash scripts/submit_iclr_spec_kv_campaign.sh
bash scripts/submit_iclr_spec_kv_two_gpu.sh
```

After stages complete, generate confidence intervals, tables, and figures with:

```bash
python paper/aggregate_campaign.py \
  --results_root outputs/iclr_spec_kv \
  --out_dir paper/campaign_artifacts
```

The strict zero-residual objective replication can be queued independently:

```bash
AFTER_JOB=<dependency-job> bash scripts/submit_kivi_no_residual_grid.sh
```

The standalone target-model quality replication is intentionally serialized
after the main campaign:

```bash
AFTER_JOB=<dependency-job> bash scripts/submit_kivi_standalone_target_quality.sh
```

The two-seed cross-family target-quality grid is serialized to one GPU and may
be dependency-gated behind the cross-family speculative campaign:

```bash
AFTER_JOB=<dependency-job> CAMPAIGN_CONCURRENCY=1 \
  bash scripts/submit_kivi_cross_family_target_quality.sh
```

The matched-geometry proposal-length ablation is also serialized and can be
placed after the cross-family role analysis:

```bash
AFTER_JOB=<dependency-job> CAMPAIGN_CONCURRENCY=1 \
  bash scripts/submit_kivi_gamma_sweep.sh
```

The targeted Qwen2.5-7B/3B confirmation should run only after the exploratory
cross-family and gamma analyses complete:

```bash
AFTER_JOB=<dependency-job> bash scripts/submit_qwen7_reversal_powered.sh
```

The expanded verifier exactness audit is serialized over one GPU:

```bash
AFTER_JOB=<dependency-job> bash scripts/submit_verifier_exactness_replication.sh
```

The actual task-accuracy study is also serialized over one GPU. Its three seeds
select disjoint examples from a common shuffled task stream:

```bash
DEPENDENCY=<dependency-job> bash scripts/submit_kivi_multiple_choice.sh
```

The missing quantizer-factorial cell is launched with:

```bash
AFTER_JOB=<dependency-job> bash scripts/submit_per_token_affine_value_grid.sh
```

Complete the ordinary-quality side of the factorial with production-style and
strict zero-residual variants:

```bash
AFTER_JOB=<dependency-job> bash scripts/submit_per_channel_symmetric_value_quality.sh
AFTER_JOB=<dependency-job> RESIDUAL_LENGTH=0 \
  bash scripts/submit_per_channel_symmetric_value_quality.sh
```

The group-size/residual-window robustness sweep is serialized cell by cell:

```bash
AFTER_JOB=<dependency-job> bash scripts/submit_kivi_group_residual_sweep.sh
```

To run only a subset of pairs:

```bash
ARRAY_RANGE=0-2 CAMPAIGN_CONCURRENCY=2 \
  bash scripts/submit_iclr_spec_kv_campaign.sh
```
