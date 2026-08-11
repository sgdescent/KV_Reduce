# KV-Cache Quantization: Collaborator Update

Status: audited results as of August 11, 2026. All speculative-decoding numbers
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

The packed prototype now has a complete systems matrix: six draft-model shapes,
batch sizes 1/4/16, contexts 1K--32K, three K/V policies, 18 source runs, and
1,512 kernel measurements. `K4V4` saves **69.26%** of cache bytes on average
and the direct Triton path is geometrically **3.43x** faster than
materialize-then-attend; `K4V8` and `K8V4` are **3.90x** and **3.61x** faster.
However, no cell beats native BF16 SDPA: geometric native-relative speeds are
0.223x, 0.315x, and 0.294x. This demonstrates real packed storage and removes
full dequantization workspace, but is not yet an end-to-end speed claim.

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
  with three disjoint seeds, 96 paired prompts per group, and metadata-adjusted
  byte accounting, is complete. The strict aggregate passes prompt-pairing,
  full-run, evaluator-version, and exact-target gates. Mild `K8V4 - K4V8` is
  unresolved at every group size. Aggressive `K4V2 - K2V4` is **-5.83 points**
  at group 16, CI **[-8.49, -3.31]**, then weakens to **-2.94**, **-1.57**, and
  **-1.87 points** at groups 32, 64, and 128, with those intervals crossing
  zero. Fine grouping therefore makes the value-precision preference strongest;
  K/V allocation and key-group geometry must be selected jointly.
- Exact BF16 recent-key residual-window sweep over 0, 32, 128, and 256 tokens
  is complete across 12 runs and passes prompt-pairing, full-run, and exact-target
  gates. For `K4V4`, no residual saves **70.56%** of draft KV and **30.87%**
  of combined KV; relative to 128 residual tokens, acceptance changes by only
  **-0.37 points**, CI **[-1.82, +1.05]**. A 256-token residual reduces draft
  savings to **62.10%** without a resolved acceptance gain. The allocation
  interaction is non-monotonic: `K8V4 - K4V8` is **+1.29 points**, CI
  **[+0.03, +2.68]**, at residual 32, while `K4V2 - K2V4` is **-3.33 points**,
  CI **[-6.26, -0.42]**, at residual 256; the other six contrasts are unresolved.
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
- Powered cross-family ARC-Challenge/HellaSwag task checks. Llama-3.2-3B and
  OLMo2-1B are complete across twelve disjoint task shards with no missing or
  underfilled runs.
  HellaSwag covers 576 examples:
  `K8V4 - K4V8` changes normalized accuracy by **-0.35 points**, CI
  **[-1.04, +0.35]**, while `K4V4` saves **58.60%** of standalone-cache bytes
  and changes accuracy by **+0.87 points**, CI **[0.00, +1.74]**. Thus, strict
  free-running token divergence does not directly imply downstream task loss.
  ARC-Challenge covers 297 examples: `K8V4 - K4V8` is **-0.67 points**, CI
  **[-2.02, +0.67]**; `K4V4` saves **53.89%** and changes accuracy by
  **+0.67 points**, CI **[-0.67, +2.02]**. For OLMo2, `K8V4 - K4V8` is
  **0.00 points**, CI **[-1.01, +1.01]**, on 297 ARC examples and **+0.17
  points**, CI **[-0.52, +1.04]**, on 576 HellaSwag examples. OLMo `K4V4`
  is within paired uncertainty of BF16 on both tasks while saving **53.91%**
  and **58.59%** of standalone cache bytes. SmolLM2 is also complete: its
  `K8V4 - K4V8` contrasts are **-0.34 points**, CI **[-1.01, 0.00]**, on
  ARC and **+0.52 points**, CI **[-0.17, +1.39]**, on HellaSwag. SmolLM
  `K4V4` exactly matches BF16 ARC accuracy and remains within uncertainty on
  HellaSwag while saving **54.62%** and **58.32%** of cache bytes. The strict
  four-model meta-aggregate rejects no summaries and finds no consistent task
  winner between `K8V4` and `K4V8`.
- The first synthetic passkey retrieval sweep is complete across 4K, 8K, and
  16K contexts, three seeds, and 72 examples per context. BF16 and every tested
  3/4/8-bit allocation achieve 100% accuracy, including keys placed at 10%,
  50%, and 90% depth. This is a ceiling result, not evidence that the policies
  are equivalent. The intended stricter execution (`13242`--`13243`) was
  correctly excluded after its summaries revealed that SLURM had captured a
  stale four-choice script; its second 100% ceiling result is not used. The
  corrected v2 sweep (`13443`--`13444`) is now complete and passes its strict
  gate: 9/9 runs report 16 choices, generator version
  `synthetic_passkey_16way_v2`, and 16 examples per run. BF16, `K4V4`,
  `K4V2`, `K2V4`, and `K2V2` all achieve 100% accuracy at 8K, 16K, and 32K,
  including passkeys placed at 10%, 50%, and 90% depth. `K2V2` saves 82.92%,
  83.26%, and 83.43% of standalone cache bytes at those contexts. This remains
  a ceiling result: increasing from four to sixteen answer choices was not
  enough to distinguish policies. The next retrieval control should use
  confusable distractors with shared prefixes, nearby values, or multi-hop
  composition rather than simply adding more choices.
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
  use a hierarchical run-cell-then-example bootstrap. Both Qwen matrices are
  now complete with **48/48 cells**, no rejected pairs, and **3,840/3,840**
  exact target outcomes each. At six bits, both objectives select top-eight
  `K4V8`; at eight bits both select `K8V8`. At the aggressive three-bit budget,
  acceptance profiling chooses a heterogeneous V policy at exactly the same
  profiled saved bytes as quality-selected `K2V4`, but is **-0.61 acceptance
  points** worse on held-out data, CI **[-1.98, +0.60]**, and has KL higher by
  **0.01485**, CI **[+0.01120, +0.02060]**. At five bits both select `K2V8`.
  OLMo2 aggressive is now also complete with **48/48 cells**, exact byte
  matching, and **3,840/3,840** exact target outcomes. At budget three, both
  objectives select top-eight `K2V4`; at budget five, both select `K2V8`, so
  cross-objective acceptance and KL effects are exactly zero. Against native
  draft KV, the selected policies change acceptance by **-0.91 points**, CI
  **[-2.22, +0.48]**, and **-0.88 points**, CI **[-2.17, +0.43]**, respectively.
  SmolLM2 aggressive is now complete with all **24/24** matrix tasks, no
  rejected pairs, and **3,840/3,840** exact target outcomes. At budget three,
  acceptance profiling improves acceptance over quality profiling by only
  **+0.11 points**, hierarchical CI **[-0.53, +0.77]**, while increasing
  quality KL by **+0.00275**, CI **[+0.00248, +0.00310]**, and delta NLL by
  **+0.00325**, CI **[+0.00107, +0.00548]**. At budget five, both objectives
  select the same policy. Across the three aggressive families, the model-
  weighted budget-three acceptance effect is **-0.17 points**, interval
  **[-0.61, +0.11]**. The current additive objective-aware allocator therefore
  has no resolved acceptance advantage.
  A strict all-layer SmolLM pilot directly compares seven uniform policies on
  32 held-out prompts/sequences and passes **224/224** target-exactness checks.
  At matched memory, `K4V2 - K2V4` is **+2.39 acceptance points**, CI
  **[-1.75, +6.73]**, but ordinary quality significantly favors `K2V4`: the
  `K4V2 - K2V4` KL contrast is **+0.03721**, CI
  **[+0.02255, +0.05279]**, and the delta-NLL contrast is **+0.05423**, CI
  **[+0.02223, +0.08759]**. These policies save roughly **71%** of draft KV and
  **12.3%** of total target-plus-draft KV. This is a promising raw preference
  reversal, not yet a resolved speculative result; powered 1K/4K, three-seed
  SmolLM, Qwen, and OLMo grids are queued as jobs `13506`--`13514`.
  The Qwen2.5 confusable 16-way passkey control is complete: nine strict runs cover
  three seeds at 8K, 16K, and 32K, with 48 paired examples per context and no
  missing or underfilled runs. Equal-memory `K4V2 - K2V4` is **+10.42 accuracy
  points** at 8K, CI **[0.00, +20.83]**; **+20.83 points** at 16K, CI
  **[+8.33, +33.33]**; and **+12.50 points** at 32K, CI
  **[+2.08, +25.00]**. At 16K, BF16 and `K4V2` score **47/48**, versus
  **37/48** for `K2V4`; at 32K the scores are **44/48**, **47/48**, and
  **41/48**, respectively. Both asymmetric policies save about **77%** of
  standalone KV. A powered 16K replication over **192 disjoint examples**
  confirms the Qwen result: `K4V2` scores **189/192 (98.44%)** versus
  **158/192 (82.29%)** for `K2V4`, a **+16.15-point** paired difference, CI
  **[+10.42, +21.88]**. BF16 scores **186/192**; the two asymmetric policies
  still save **77.06%** and **77.01%** of standalone KV. This is synthetic
  draft-only retrieval-quality evidence, not speculative acceptance. A matched
  strict Llama-3.2-3B replication reverses
  the resolved 32K preference: `K4V2 - K2V4` is **-10.42 points**, CI
  **[-18.75, -2.08]**, with **42/48** versus **47/48** correct. Its 8K and
  16K contrasts are unresolved. A powered Llama 16K replication is also
  unresolved: `K4V2` scores **185/192** versus **183/192** for `K2V4`, a
  **+1.04-point** contrast, CI **[-1.56, +4.17]**. Thus, even for the same task and quantizer,
  retrieval precision preference is model-dependent. Qwen3-4B is now complete
  across all nine strict runs: every policy scores **48/48** at 8K and 16K; at
  32K, `K4V2` scores **47/48** and `K2V4` **48/48**, an unresolved **-2.08
  point** contrast, CI **[-6.25, 0.00]**. Across Qwen2.5, Llama 3, and Qwen3,
  model-bootstrap macro `K4V2 - K2V4` effects are **+4.86 points** at 8K, CI
  **[0.00, +10.42]**; **+6.25 points** at 16K, CI **[-2.08, +20.83]**; and
  exactly **0.00 points** at 32K, CI **[-10.42, +12.50]**. The cross-family
  aggregate therefore rejects a universal K-first or V-first retrieval policy;
  key/value allocation must be model- and context-aware. The quantizer-axis
  ablation (`13538`--`13540`) is now complete and resolved. On the same 48
  Qwen2.5 16K examples, per-channel keys yield a `K4V2 - K2V4` contrast of
  **+20.83 points**, CI **[+8.33, +33.33]**, while per-token keys yield
  **+75.00 points**, CI **[+62.50, +87.50]**. The axis-by-allocation interaction
  is **-54.17 points** for per-channel minus per-token, CI
  **[-72.92, -35.42]**. Per-token `K2V4` retrieves only **4/48**, versus
  **40/48** for `K4V2`. This shows that K/V sensitivity must be reported jointly
  with quantizer geometry; per-channel KIVI-style keys preserve critical
  channel-wise outliers that per-token keys destroy.
  The OLMo2 mild replication (`13386`--`13390`) is now complete with all
  **24/24** held-out cells, exact allocation-byte matching, and **3,840/3,840**
  exact target outcomes. Quality and acceptance profiling choose the same
  top-eight `K4V8` layout at budget six and `K8V8` at budget eight, making all
  cross-objective acceptance/KL/NLL contrasts exactly zero. Against native KV,
  the selected policies change acceptance by only **+0.06 points**, hierarchical
  CI **[-0.32, +0.47]**, and **+0.11 points**, CI **[-0.20, +0.43]**. The
  cross-context `K8V4 - K4V8` heuristic contrast is **-0.41 acceptance points**,
  CI **[-1.01, +0.16]**, while `K8V4` incurs **+0.00140 KL**, CI
  **[+0.00092, +0.00214]**. This is a clean mild-budget null/control rather than
  evidence for a universal K-first policy. The SmolLM2 mild chain
  (`13391`--`13395`) is now running with the same 24-cell strict design.
- Larger-pair exact-byte replications are dependency-queued after those stages.
  Qwen2.5-7B/3B aggressive (`13398`--`13402`) and mild
  (`13403`--`13407`) matrices are followed by Llama-3.1-8B/3.2-3B aggressive
  (`13408`--`13412`) and mild (`13413`--`13417`) matrices. Target and draft
  occupy separate GPUs; each 24-cell array is capped at two concurrent
  two-GPU tasks, preserving the four-GPU campaign ceiling.
- The powered all-layer SmolLM2 replication is complete across 1K and 4K,
  three seeds per context, **288 prompts**, and **2,016/2,016** exact target
  trajectories. Equal-memory `K4V2 - K2V4` acceptance is **-1.34 points** at
  1K, CI **[-3.24, +0.54]**, and **+0.63 points** at 4K, CI
  **[-1.15, +2.49]**. Ordinary-quality KL decisively favors `K2V4`: the same
  contrast is **+0.03128** at 1K, CI **[+0.02541, +0.03767]**, and
  **+0.04353** at 4K, CI **[+0.03154, +0.05935]**. The 4K means reverse, but
  acceptance remains unresolved after powering the test.
- Direct all-layer replications remain dependency-queued for Qwen2.5,
  OLMo2, and Llama (`13506`--`13517`). A strict four-family meta-analysis
  (`13518`) will run only after all four aggregates pass completeness, evaluator,
  and target-exactness gates.
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
