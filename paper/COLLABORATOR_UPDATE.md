# KV-Cache Quantization: Collaborator Update

Status: audited results as of August 10, 2026. All speculative-decoding numbers
below use a sequential BF16 target verifier, an unquantized target cache, and
produce exactly the same target-greedy tokens as the independent BF16 target.

## Copy-Paste Update

We have pivoted from cross-model KV translation to KV-cache quantization. This
should **not** be restricted to speculative decoding: ordinary autoregressive
LLMs also benefit because every decoding step reads the growing KV cache. The
difference is correctness. Quantizing a standalone model's cache can alter all
future tokens, so it must preserve NLL/KL, free-running generations, retrieval,
and downstream tasks. Quantizing only the draft cache in speculative decoding
changes proposal quality and speed, but exact BF16 target verification preserves
the target output distribution.

Quantization alone is not enough novelty for a main-track paper. KIVI already
establishes asymmetric K/V quantization, KV-AdaQuant allocates different K/V
precision, RateQuant optimizes quantizer-specific mixed precision, Block-GTQ
allocates precision within RoPE key blocks, and QuantSpec applies quantized KV
caches to speculative decoding.
The August 2026 NVIDIA work on cross-model KV transfer also substantially
occupies our earlier closed-form cache-mapping direction. That paper does not,
however, study byte-matched precision allocation under ordinary-quality versus
speculative-acceptance objectives.
Our possible main-track contribution is narrower and more controlled:

> Quantizer geometry, model architecture, and downstream decoding objective
> jointly determine the best KV precision allocation; allocation must therefore
> be optimized at equal actual bytes and validated on the metric used in serving.

The strongest exact speculative result now covers six target/draft pairs from
Qwen2.5, Qwen3, Llama 3, OLMo 2, and SmolLM2. Across 18 runs, 576 prompts, and
2,304 quantization trajectories, draft `K4V4` removes **66.13%** of draft-cache
bytes and **22.64%** of combined target-plus-draft KV. Its model-weighted
acceptance change is **-0.79 percentage points** with 95% hierarchical-bootstrap
CI **[-1.63, +0.04]**. The equal-mean-bit comparison `K8V4 - K4V8` is
**-0.05 points**, CI **[-0.76, +0.63]**. Thus, mild key-versus-value allocation
is statistically unresolved across these six pairs; we should not claim that
keys universally require more bits.

The aggressive equal-mean-bit control is now complete across the same six
families and three seeds per pair. Over 576 paired prompts, `K4V2 - K2V4`
changes acceptance by **-1.94 points**, CI **[-3.50, -0.40]**. Thus, under
KIVI-style grouped per-channel key quantization, allocating the extra two bits
to values is significantly better at this aggressive budget.

The most novel completed mechanism result is a controlled geometry reversal.
With per-token key quantization, `K8V4 - K4V8` changes acceptance by
**+28.70 points**; changing only the key quantization axis to KIVI-style grouped
per-channel quantization changes the same contrast to **-0.33 points**. At the
more aggressive two-bit setting, `K4V2 - K2V4` reverses from **+15.05** to
**-3.66 points**. This shows that Gaussian-noise sensitivity or results from one
quantizer cannot establish a universal K/V precision rule.

Ordinary free-running decoding provides an independent control. For
Qwen2.5-1.5B with a 1K prefix and 256 generated tokens, `K4V8` retains 33.16%
of BF16 tokens versus 23.57% for `K8V4`; the paired difference
`K8V4 - K4V8` is **-9.59 points**, CI **[-18.98, -0.47]**. Under the tested
KIVI geometry, preserving value precision can therefore matter more during
long free-running generation even though mild speculative-acceptance differences
are unresolved. Three-seed 256-token replications are now complete for three
additional families. The `K8V4 - K4V8` token-retention contrast is -11.28
points for OLMo, CI [-22.52, -0.26]; -4.96 for SmolLM2, CI [-17.08, +7.29];
and +0.11 for Llama, CI [-9.87, +10.31]. The policy is therefore model-dependent
rather than universal. Their equal-model macro contrast is -5.38 points,
CI [-13.72, +2.78]. Across these three models, `K4V4` saves 66.56% of
standalone-cache bytes and retains 28.68% of exact BF16 tokens over 256-token
continuations. Token retention is a strict trajectory-drift diagnostic, not a
semantic-quality score; downstream tasks remain necessary.

The aggressive 256-token replication is also complete for Llama, OLMo, and
SmolLM: 9 full runs and 144 paired prompts. `K4V2 - K2V4` changes the retained
BF16-prefix fraction by **-2.74 points**, CI **[-6.02, -0.94]**, significantly
favoring value precision. The full-continuation token-agreement contrast is
**-1.71 points**, CI **[-4.47, +0.49]**, and remains unresolved. In this
aggressive sweep, `K4V4` saves **66.56%** of standalone-cache bytes and retains
**24.67%** exact BF16 tokens over 256-token continuations.

This is promising but not yet sufficient for a main-track method paper. The
remaining gates are a held-out geometry- and objective-aware allocator that
beats uniform/asymmetric baselines at equal actual bytes, and a packed attention
implementation that converts compression into measured long-context throughput
or batch-capacity gains. If those gates fail, the work remains a useful empirical
study, but the main-track novelty case is substantially weaker.

## Exact Speculative Results

All values are model-pair-level token acceptance rates. Deltas are `K4V4 - BF16
draft`; cache savings include quantization metadata and a BF16 residual tail.

| Target / draft | BF16 | K4V4 | Delta, 95% CI (pp) | Draft KV saved | Total KV saved |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-3B / 1.5B | 59.78% | 59.02% | -0.76 [-2.23, +0.66] | 66.33% | 29.02% |
| OLMo2-7B / 1B | 54.03% | 53.17% | -0.86 [-1.98, +0.18] | 66.33% | 13.27% |
| SmolLM2-1.7B / 360M | 52.95% | 51.10% | -1.85 [-3.21, -0.54] | 65.55% | 11.30% |
| Qwen2.5-7B / 3B | 61.80% | 62.08% | +0.28 [-0.98, +1.47] | 66.33% | 25.96% |
| Qwen3-8B / 4B | 54.06% | 52.39% | -1.67 [-3.12, -0.30] | 65.87% | 25.33% |
| Llama3.1-8B / Llama3.2-3B | 64.95% | 65.09% | +0.13 [-1.27, +1.60] | 66.33% | 30.96% |
| Equal-model macro | 57.93% | 57.14% | -0.79 [-1.63, +0.04] | 66.13% | 22.64% |

Interpretation:

- `K4V4` preserves acceptance within uncertainty on four of six model pairs.
- SmolLM2 and Qwen3 have small but statistically resolved acceptance losses.
- Combined savings depend on how large the draft cache is relative to the target
  cache; draft-only quantization cannot remove the unquantized target cache.
- Every quantized trajectory exactly matches target-greedy output. The cost of
  draft quantization is lower acceptance, not incorrect final tokens.

## Why Ordinary Decoding Matters

The same cache quantizer supports three deployment regimes:

1. **Standalone autoregressive decoding:** quantize the model's cache directly.
   This reduces memory and HBM reads but can compound errors after the first
   changed token. Evaluate NLL/KL, token retention, task accuracy, retrieval,
   generation quality, throughput, and peak memory.
2. **Draft-only speculative decoding:** quantize only the draft cache and keep
   the target/verifier BF16. Evaluate acceptance, accepted tokens per target
   call, throughput, and combined cache bytes. Exact verification preserves the
   final target distribution.
3. **Joint target/draft quantization:** compress both caches for larger savings.
   This is approximate because the verifier itself changes, so it needs the same
   quality audits as standalone decoding and cannot claim exact target output.

The scientific comparison is whether policies selected for ordinary quality and
speculative acceptance differ at the same actual byte budget. Our current
evidence does **not** establish a universal objective-specific bit reversal:
ordinary KL is often a strong risk screen, while direct acceptance remains the
correct final metric for speculative serving.

## Main-Track Assessment

**Current verdict:** plausible direction, insufficient novelty today if framed
as simply “quantize KV for speculative decoding.”

A credible main-track submission should contain:

- A controlled finding that survives models, tasks, contexts, quantizer axes,
  group sizes, residual-window sizes, and multiple seeds.
- A byte-constrained allocator that selects per-layer/per-component geometry and
  bit-width using held-out calibration, and outperforms KIVI, uniform precision,
  KV-AdaQuant-style K/V allocation, and QuantSpec-style baselines.
- Exact speculative evaluation and ordinary free-running/task evaluation under
  the same cache formats, with paired confidence intervals.
- Packed/fused kernels with real cache storage, no full dequantization workspace,
  and measured long-context throughput or batch-capacity gains.

The packed prototype already demonstrates feasibility: a Qwen2.5-1.5B-shaped
`K4V4` cache at 32K drops from 896 MiB to 260.2 MiB (**70.96% saved**), and the
direct Triton path is 2.05x faster than materialize-then-attend while using only
0.38 MiB of temporary memory. It is still 6.88x slower than native BF16 SDPA,
so it is not yet an end-to-end speed claim.

## Work In Progress

- Exact proposal-length sweep at `gamma = 2, 4, 8`, including aggressive
  `K4V2` and `K2V4` controls. All three proposal lengths are complete across
  three seeds. `K4V2 - K2V4` is **-5.04**, **-4.80**, and **-3.55** acceptance
  points at gamma 2, 4, and 8,
  respectively; all three confidence intervals exclude zero. The mild
  `K8V4 - K4V8` contrast is unresolved at gamma 2 and 4 but is **+1.42
  points**, CI **[+0.35, +2.52]**, at gamma 8.
- Six-pair exact `K4V2` versus `K2V4` cross-family replication, paired with an
  ordinary 256-token free-generation control on Llama, OLMo, and SmolLM. All 18
  exact runs are complete: three seeds for each of six target/draft pairs, 576
  paired prompts, and 1,728 quantized trajectories. Across the six pairs,
  `K4V2 - K2V4` is **-1.94 acceptance points**, CI **[-3.50, -0.40]**.
  The model-weighted direction significantly favors preserving values under the
  tested KIVI geometry.
- Exact key-quantization group-size sweep over 16, 32, 64, and 128 channels,
  with three disjoint seeds and metadata-adjusted byte accounting. An audit
  found that the first array version advanced the FineWeb offset by treatment,
  so it is excluded from between-group conclusions. The corrected prompt-paired
  rerun is `13249`--`13251`; a strict meta-aggregate verifies identical offsets,
  full rows, and exact target outputs before computing paired intervals.
- Exact BF16 recent-key residual-window sweep over 0, 32, 128, and 256 tokens
  to separate low-bit compression from the protection of recent context. Its
  corrected prompt-paired chain is `13252`--`13254` and uses the same strict
  treatment-pairing gate.
- The paper-grade PG19-train long-context extension (`13218`--`13219`) is
  complete: six full runs, three seeds per context, 60 prompts, and 360 exact
  target-matching trajectories at 16K and 32K. `K4V4` saves **30.99%** of
  combined target-plus-draft KV at 16K with an acceptance change of **-0.94
  points**, CI **[-2.06, -0.13]**; at 32K it saves **31.04%** with a **-0.73
  point** change, CI **[-2.99, +1.28]**. The mild allocation contrast
  `K8V4 - K4V8` is unresolved at 16K (**+0.13 points**, CI
  **[-1.09, +1.26]**) but favors preserving values at 32K (**-3.50 points**,
  CI **[-6.62, -0.45]**). The aggressive contrast remains unresolved at both
  contexts. Earlier sparse 4K/8K results remain directional controls, and the
  underfilled first-generation 16K cell stays excluded.
- Matched ordinary-LM PG19-train quality and exact acceptance are complete at
  the same 16K/32K windows, seeds, K/V policies, and quantizer geometry
  (`13229`--`13232`). There is **no statistically resolved objective-preference
  reversal**. Quality KL favors `K4V8` over `K8V4` at 16K by **0.00305**, CI
  **[0.00210, 0.00427]**, and at 32K by **0.00309**, CI
  **[0.00203, 0.00429]**; exact acceptance agrees at 32K and is unresolved at
  16K. Quality also strongly favors `K2V4` over `K4V2`, while acceptance is
  unresolved at both contexts. Spearman correlation between acceptance harm
  and quality KL across the five policies is **0.60 at 16K** and **0.80 at
  32K**. KL is therefore a useful screen here, but direct acceptance remains
  the final serving objective. The per-layer byte-matched matrices are the
  stronger remaining test of objective-specific allocation.
- Qwen2.5-3B and Qwen3-4B ordinary free-running size sweep; all twelve cells
  are complete. The 64-token sweep covers three seeds per
  model: `K8V4 - K4V8` changes exact-token retention by **-5.12 points**, CI
  **[-11.22, +0.77]**, while the retained-prefix contrast is **-8.23 points**,
  CI **[-15.22, -1.31]**. The 256-token sweep gives an
  exact-token contrast is **-7.52 points**, CI **[-17.85, +1.39]**. At 256
  tokens, `K4V4` saves **66.59%** of standalone-cache bytes and retains
  **25.10%** of exact BF16 tokens. The separate three-seed
  SmolLM2/OLMo2/Llama replication is also complete.
- Powered cross-family ARC-Challenge/HellaSwag task checks. Llama-3.2-3B
  is complete across six disjoint task shards with no underfilled runs.
  HellaSwag covers 576 examples:
  `K8V4 - K4V8` changes normalized accuracy by **-0.35 points**, CI
  **[-1.04, +0.35]**, while `K4V4` saves **58.60%** of standalone-cache bytes
  and changes accuracy by **+0.87 points**, CI **[0.00, +1.74]**. Thus, strict
  free-running token divergence does not directly imply downstream task loss.
  ARC-Challenge covers 297 examples: `K8V4 - K4V8` is **-0.67 points**, CI
  **[-2.02, +0.67]**; `K4V4` saves **53.89%** and changes accuracy by
  **+0.67 points**, CI **[-0.67, +2.02]**. OLMo2 and SmolLM2 task shards are
  running.
- The first synthetic passkey retrieval sweep is complete across 4K, 8K, and
  16K contexts, three seeds, and 72 examples per context. BF16 and every tested
  3/4/8-bit allocation achieve 100% accuracy, including keys placed at 10%,
  50%, and 90% depth. This is a ceiling result, not evidence that the policies
  are equivalent. A stricter 16-choice replacement (`13242`--`13244`)
  evaluates `K4V2`, `K2V4`, and `K2V2` at 8K/16K/32K with three disjoint
  seeds and a completeness gate that rejects missing or underfilled runs.
- Exact quality-optimized versus acceptance-optimized allocation matrix at
  actual packed bytes: two budgets, 1K/4K contexts, three held-out seeds, and
  24 ordinary/speculative cross-evaluation cells per family. An audit found
  that the first queued version matched nominal mean bits rather than metadata-
  aware bytes and omitted 2-bit candidates during aggressive allocation; those
  jobs were canceled before using GPU time. Corrected chains are Qwen mild
  (`13264`--`13268`), Qwen aggressive (`13269`--`13273`), OLMo2 aggressive
  (`13274`--`13278`), and SmolLM2 aggressive (`13279`--`13283`). Quality and
  acceptance profiles now use identical sequence lengths and must achieve the
  same profiled saved-byte target. The final aggregator fails closed on byte
  mismatch, missing or underfilled cells, wrong evaluator versions, non-exact
  target trajectories, or incomplete policy coverage; cross-context intervals
  use a hierarchical run-cell-then-example bootstrap.
- Model-weighted paper tables and objective-comparison figures.
- Packed-kernel optimization and a serving-capacity benchmark.

## Closest Work

- [KIVI](https://arxiv.org/abs/2402.02750)
- [AsymKV](https://arxiv.org/abs/2410.13212)
- [KV-AdaQuant](https://arxiv.org/abs/2502.15075)
- [QuantSpec](https://arxiv.org/abs/2502.10424)
- [RotateKV](https://arxiv.org/abs/2501.16383)
- [RateQuant](https://arxiv.org/abs/2605.06675)
- [RoPE-Aware Bit Allocation](https://arxiv.org/abs/2606.24033)
- [Adaptive KV-Cache Quantization](https://arxiv.org/abs/2604.04722)
- [Cross-Model KV Cache Transfer](https://arxiv.org/abs/2608.03893)
