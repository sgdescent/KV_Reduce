# KV-Cache Quantization: Collaborator Update

Status: provisional results as of August 10, 2026. The matched-objective grid,
three-length speculation sweep, 16K/32K speculative long-context sweep,
four-condition verifier audit, corrected Qwen HellaSwag/ARC task suite, and
powered Qwen2.5-7B/3B FineWeb-Edu K4V3-versus-K3V4 replication are complete.
The first real bit-packed memory/attention benchmark is also complete.
Cross-family task checks, all-layer objective-specific allocation, strict
quantizer-factorial cells, long-context replications, and free-running
standalone-generation drift are still running.

## Bottom Line On Scope And Novelty

KV-cache quantization is useful for ordinary autoregressive LLMs as well as
speculative decoding. In ordinary decoding, compressed K/V reduces the cache
that every decode step reads, which can increase maximum context length and
batch capacity; because the compressed cache directly drives token generation,
we must measure KL/NLL, task accuracy, and long-generation drift. In draft-only
speculative decoding, the same compression reduces draft-cache memory and
bandwidth, but an exact BF16 target verifier corrects the proposals, so cache
error changes acceptance and speed rather than the final target distribution.

Quantization by itself is not sufficient novelty for a main-track paper. Our
main-track case depends on delivering all three pieces together: (1) the
controlled finding that quantizer geometry, not only bit-width, determines the
apparent K/V sensitivity; (2) a byte-constrained, layer-wise allocator calibrated
for ordinary quality and speculative acceptance; and (3) a packed implementation
that converts those policies into measured long-context memory-capacity or
throughput gains. We already have strong evidence for the first piece and broad
evidence that K4V4/K3V4 preserve behavior. The allocator, cross-family power,
and packed systems results remain the gates for a credible main-track claim.

The prior-work bar is high. KIVI already establishes asymmetric quantization
geometry for ordinary decoding, KV-AdaQuant explicitly assigns different
precision to K and V, and QuantSpec studies quantized caches inside speculative
decoding. Our differentiator therefore cannot be merely "use different K/V
bits" or "quantize the draft cache." It must be the controlled geometry result,
objective-aware byte allocation, and a measured packed implementation evaluated
under both ordinary and speculative decoding.

## Direct Answers For The Team

**Is this sufficient novelty for a main-track paper?** Not yet as a method claim.
The empirical observation is interesting, but the area already contains KIVI,
KV-AdaQuant, QuantSpec, and other adaptive KV quantizers. It becomes a plausible
main-track paper if we show that prior sensitivity conclusions are confounded by
quantizer geometry, introduce a robust layer-wise allocator that beats uniform
and published asymmetric policies at equal *actual bytes*, and demonstrate a
real serving benefit with packed/fused attention. The current evidence supports
the first part; the campaign is testing the second, and the packed benchmark has
validated memory storage but not production speed.

**Why restrict this to speculative decoding?** We should not. In ordinary
autoregressive decoding, quantizing the model's K/V cache can reduce memory
traffic, increase context length, and increase batch capacity. The tradeoff is
that errors directly alter future tokens and can compound through generation.
In draft-only speculative decoding, quantization errors only change proposal
quality and acceptance: exact target verification still preserves the target
distribution. This makes speculative decoding a safety envelope and a clean
measurement setting, while ordinary generation is a first-class deployment
target and an important control.

**What evidence is available now?** The actual packed K4V4 cache for a
Qwen2.5-1.5B-shaped GQA configuration reduces persistent storage from 896 MiB
to 260.2 MiB at 32K tokens, a 70.96% reduction including metadata and a BF16
residual tail. Savings are already 66.80% at 1K and approach 71% as metadata is
amortized. The current materialize-then-attend CUDA diagnostic is 14.5x slower
than native attention at 32K, so it validates the byte accounting but also shows
that a fused packed attention kernel is mandatory before making a speed claim.
Separately, a free-running target-cache campaign is queued across Qwen, Llama,
OLMo, and SmolLM to measure exact sequence retention, token agreement, first
divergence, and long-horizon error accumulation outside speculative decoding.

## Copy-Paste Message

The cache-quantization pivot is promising, but cache quantization alone is not
yet enough novelty for a main-track paper. KV-cache quantization, asymmetric K/V
precision, and quantized speculative decoding already have strong prior work.
Our completed matched-objective grid also rejects the strongest version of our
initial hypothesis: speculative acceptance harm and ordinary-LM KL are strongly
rank-correlated at 1K and 4K (Spearman 0.964 and 0.878), with no statistically
resolved equal-memory preference reversal. We should not claim that speculative
decoding universally needs a different K/V bit allocation.

The sharper result is instead that *quantizer geometry controls the apparent K/V
sensitivity*. Holding model, prompts, value quantizer, and nominal bit budget
fixed, changing only keys from per-token to grouped per-channel quantization
moves the K8V4-minus-K4V8 acceptance contrast from +28.70 to -0.33 points and
reverses the two-bit K/V preference. A credible main-track story would combine
this controlled mechanism result with a geometry-aware allocator and packed
kernels that improve long-context memory, batch capacity, or throughput over
KIVI, uniform precision, and QuantSpec-style baselines.

This should not be restricted to speculative decoding. The same cache quantizer
applies to ordinary autoregressive LLMs, where it can reduce cache memory and
increase serving capacity. We are evaluating one quantization policy under three
deployment regimes: ordinary autoregressive decoding, draft-only speculative
quantization, and joint target/draft quantization. The important distinction is
correctness: draft-only quantization remains distribution-exact because the BF16
target verifier corrects every proposal, whereas quantizing a standalone model or
the target cache changes the output distribution and therefore requires direct
KL, NLL, top-1, task-accuracy, and generation-quality evaluation.

The current main-track opportunity is therefore broader than "quantization for
speculative decoding." It is a controlled account of how quantizer geometry and
downstream objective determine K/V precision, together with a geometry-aware
allocator and, ultimately, packed kernels that translate the chosen policies into
long-context memory-capacity or throughput gains. The speculative setting gives
us an unusually clean systems objective---acceptance---and an exact verifier, but
ordinary decoding is both a control and a first-class deployment target.

The strongest validated numbers so far are encouraging. Draft K4V4 saves 66.13%
of draft-cache storage and 22.64% of combined target-plus-draft KV with a +0.10
point macro acceptance change (95% CI: -0.04 to +0.25). Joint target K4V8 plus
draft K8V4 reaches 54.59% combined KV savings at 1K with a -0.46 point acceptance
change (CI: -1.78 to +0.84), but it is approximate and does not yet satisfy our
conservative acceptance bound at 4K. The exact-target alternative keeps target
BF16 and quantizes draft K4V4, saving 29.02% at 1K and 30.58% at 4K while retaining
the target distribution. The campaign is still testing longer contexts, tasks,
speculation lengths, and all-layer objective-specific allocations.

The new powered Qwen2.5-7B/3B FineWeb-Edu result reinforces the geometry-aware
interpretation. Across 1,526 paired speculative prompts, K3V4 beats equal-memory
K4V3 by 0.65 acceptance points (95% CI: 0.15 to 1.16). K3V4 itself is within
0.06 points of native acceptance (CI: -0.46 to +0.34) while saving 69.03% of the
draft cache and 27.01% of combined KV. Ordinary quality independently favors
K3V4: K4V3 increases KL by 0.00848 (CI: 0.00777 to 0.00927). This rejects the
exploratory objective reversal; under the tested KIVI-style geometry, preserving
value precision is better for both ordinary and speculative decoding.

## Short Update To Share

We have pivoted from cross-model cache translation to objective-aware KV-cache
quantization. The central question is not merely whether a KV cache can be
quantized, but whether the precision allocation should change with the
downstream decoding objective. We compare ordinary language-model quality
(KL, continuation NLL, top-1 agreement, and downstream tasks) against
speculative-decoding utility (acceptance rate, accepted tokens per verification,
and acceptance mass) under the same model, prompts, quantizer, and memory budget.

Our current two-seed aggregate covers six target/draft pairs from Qwen2.5,
Qwen3, Llama 3, OLMo 2, and SmolLM2. With the KIVI-style quantizer geometry,
uniform K4V4 removes 66.13% of the draft KV cache and 22.64% of combined
target-plus-draft KV memory. Its macro speculative-acceptance change is +0.10
percentage points (95% CI: -0.04 to +0.25), with ordinary-LM KL 0.00730 and
95.99% top-1 agreement. K4V8 is more conservative: it removes 53.63% of the
draft cache and 18.36% of total KV memory, with a +0.41 point acceptance change
(95% CI: +0.03 to +0.84), KL 0.00190, and 97.80% top-1 agreement. We treat the
small positive acceptance deltas as preservation, not as evidence that
quantization improves the model.

The clearest result so far is that quantizer geometry changes the apparent K/V
sensitivity. We paired the same six model families at 1K context. With naive
per-token symmetric quantization, K8V4 beats equal-memory K4V8 by 10.87
acceptance points on average (95% model-pair bootstrap CI: 3.75 to 19.68) and
wins five of six pairs. With KIVI-style grouped per-channel affine keys, the
same contrast is -0.25 points (CI: -0.56 to +0.04) and K8V4 wins only two of
six. The paired geometry shift is -11.12 points (CI: -19.71 to -4.25). Thus,
the earlier conclusion that keys inherently require more precision was largely
an artifact of applying a poor quantization axis to persistent key-channel
outliers.

The controlled three-seed factorial result is even sharper. Holding affine value
quantization, model, prompts, and nominal bit budget fixed, per-token keys make
K8V4 beat K4V8 by 28.70 acceptance points across 186 paired prompts (95% CI:
26.10 to 31.38). Changing only the key axis to grouped per-channel quantization
makes the same contrast -0.33 points across 189 prompts (CI: -1.26 to 0.62).
At two bits, K4V2-minus-K2V4 reverses from +15.05 points (CI: 13.58 to 16.65)
to -3.66 points (CI: -5.80 to -1.56). Ordinary-quality KL independently flips
in the same direction: the K8V4-minus-K4V8 KL contrast changes from -2.023 to
+0.00440. This establishes a controlled geometry-induced preference reversal,
not merely a cross-family correlation.

Within the KIVI geometry, reducing value precision from four to three bits
(K4V3) is more harmful on average than reducing key precision (K3V4): K4V3
changes acceptance by -1.01 points and has KL 0.0327, while K3V4 changes
acceptance by -0.32 points and has KL 0.0111. Gaussian perturbation sensitivity
therefore cannot be treated as a direct proxy for quantization sensitivity.

The completed three-seed proposal-length sweep resolves this asymmetry at two bits.
Across 191 paired prompts, K4V2 loses 5.43 acceptance points from BF16 (95% CI:
-7.32 to -3.58), while K2V4 loses 1.43 points (CI: -2.58 to -0.31). The direct
paired K4V2-minus-K2V4 contrast is -4.01 points (CI: -5.82 to -2.15). The two
policies have the same nominal mean bit-width and near-equal estimated total-KV
savings (31.76% versus 31.38%); the small byte difference comes from asymmetric
key metadata. This is evidence that, with grouped per-channel KIVI keys,
aggressive value quantization can be more harmful than aggressive key
quantization. This replicates at `gamma=4` across 189 paired prompts: K4V2
loses 6.48 points (CI: -8.39 to -4.61), K2V4 loses 2.41 points
(CI: -3.83 to -0.98), and the direct contrast is -4.07 points
(CI: -5.95 to -2.26). At `gamma=8`, the direct contrast remains -3.08 points
across 191 paired prompts (CI: -4.70 to -1.48). The sign and statistical
conclusion are therefore stable across proposal lengths 2, 4, and 8.

The direct 20-configuration objective grid is now complete at 1K and 4K with
three seeds per context. Acceptance harm and ordinary-quality KL have Spearman
correlations 0.964 and 0.878, respectively, and none of six near-memory-matched
comparisons reverses preference. K4V4 is the maximum-savings configuration that
meets the predeclared two-point acceptance and 0.01-KL budgets at both contexts,
saving 29.02% of combined target-plus-draft KV at 1K and 30.58% at 4K. This is a
useful negative result: ordinary quality is a strong screening objective, while
direct acceptance remains the final systems metric rather than a proven source
of a different bit policy.

A CPU-only calibration-size rehearsal on an older all-layer WikiText profile
shows why the new FineWeb held-out campaign is necessary. Relative to a
16-prompt profile, the 8-prompt acceptance-risk ranking has Spearman 0.852 and
identical top-8 sensitive cells, but its final acceptance allocation still
changes 25% of K/V layer decisions. The ordinary-quality allocation is much
more stable, changing only 3.6% of decisions. This is a pipeline diagnostic,
not paper evidence: the old profile predates the current FineWeb data protocol,
and its aggressive 3-bit allocations collapse held-out speculative acceptance.
The fresh campaign therefore uses 16 calibration examples, an 8-bit mean
budget, a uniform matched-budget baseline, and disjoint held-out evaluation.

The speculative PG19 sweep is also complete. At 16K, K4V4 saves 70.79% of the
draft cache and 30.97% of combined KV with a +0.30-point acceptance change
(CI: -2.25 to +3.04) across 24 paired prompts. At 32K it saves 70.94% of the
draft cache and 31.04% combined KV; the -2.22-point estimate has a wide interval
(-5.81 to 0.00) over only 11 prompts, so we treat 32K speculative acceptance as
preliminary. The matched ordinary-quality sweep is complete across three seeds:
K4V4 has KL 0.00624 at 16K and 0.00724 at 32K, with 96.35% and 96.88% top-1
agreement while removing 70.79% and 70.94% of standalone KV. K4V8 has
significantly lower KL than equal-budget K8V4 at both contexts, agreeing with the
speculative preference at 16K rather than producing an objective reversal.

One exploratory Qwen2.5-7B/3B seed initially suggested an equal-memory objective
reversal: speculative acceptance favored K4V3 over K3V4 by +0.43 points while
ordinary quality favored K3V4. The completed three-seed WikiText diagnostic does
not replicate that sign. K4V3-minus-K3V4 acceptance is -0.64 points across 763
paired prompt occurrences, with a nominal 95% interval of [-1.29, -0.001], so
both objectives currently favor K3V4. The ordinary-quality KL contrast is
+0.01294 (95% CI: +0.01194 to +0.01396) across 750 sequence occurrences, also
making K4V3 worse. These intervals are diagnostic only: every speculative run
requested 512 prompts but the validation split supplied exactly 256 usable 1K
blocks, and every quality run supplied 250 of 256 requested sequences. Seeds
therefore reshuffle a finite pool rather than form independent samples. We have
replaced this with a streaming FineWeb-Edu replication using three explicit,
non-overlapping shards of 512 speculative prompts and 256 ordinary-quality
sequences per shard. Aggregators report every requested-versus-observed sample
shortfall explicitly.

All three powered FineWeb-Edu shards are now complete. The ordinary-quality arm
covers 768 requested and observed sequences with disjoint streaming offsets.
K4V3 has mean KL 0.01529 versus 0.00681 for K3V4. The paired
K4V3-minus-K3V4 KL contrast is +0.00848 (95% bootstrap CI: +0.00777 to
+0.00927), and K4V3 has 1.86 points lower top-1 agreement. Thus, ordinary
quality decisively favors preserving value precision.

The powered speculative arm covers three disjoint 512-prompt shards. In the
direct K4V3-versus-K3V4 comparison, 1,526 valid paired prompt occurrences give
an acceptance contrast of -0.65 points (95% CI: -1.16 to -0.15), again favoring
K3V4. Relative to native BF16 draft caches, K3V4 changes acceptance by only
-0.06 points (95% CI: -0.46 to +0.34), while K4V3 loses 0.71 points (95% CI:
-1.19 to -0.24). K3V4 saves 69.03% of draft-cache bytes and 27.01% of combined
target-plus-draft KV. The powered result therefore rejects both the original
objective-reversal hypothesis and a geometry-independent claim that keys always
require more precision.

The broader cross-family result suggests a practical two-stage policy even
without a resolved preference reversal. Across 48 model-pair/configuration cells,
ordinary-quality KL and speculative-acceptance harm have Spearman correlation
0.849; the model-pair macro average is 0.794 (95% bootstrap CI: 0.631 to 0.929).
A leave-one-model-pair-out linear predictor using only ordinary-quality log-KL and
top-1 agreement predicts acceptance harm with 1.01-point RMSE and R-squared 0.724.
A conservative KL <= 0.01 gate selects 19 cells, and all 19 remain within a
two-point acceptance-loss budget, although it recovers only 54.3% of all safe
cells. Thus, ordinary-LM evaluation can cheaply reject risky configurations;
acceptance evaluation is still needed to identify additional aggressive but safe
operating points. This is an empirical screen on the tested cells, not a formal
guarantee.

We also completed the target-versus-draft role comparison. Across the same six
model pairs and eight K/V precision settings, the larger target checkpoint has
lower ordinary-LM KL than the smaller draft in all 48 matched comparisons. At
K4V4, target KL is 0.00571 versus 0.00730 for the draft (paired macro difference
-0.00159, 95% CI: -0.00276 to -0.00068); at K4V8, it is 0.00157 versus 0.00190
(difference -0.00033, CI: -0.00046 to -0.00021). Since target role and model size
are confounded, this supports role-and-scale-aware calibration rather than a
causal claim that being a target makes a model robust. Quantizing the target also
changes the final model distribution, so its policy needs stricter top-1 and task
quality constraints than draft-only quantization.

The synthetic passkey audit is also complete across 4K, 8K, and 16K contexts,
three insertion depths, and three disjoint seeds. BF16 and every tested
quantization policy achieve 100% accuracy on 216 examples per policy. K4V4 saves
69.96%, 70.53%, and 70.81% of standalone KV storage at 4K, 8K, and 16K,
respectively. This rules out an obvious retrieval failure, but the task is
saturated and cannot rank precision allocations; we therefore treat it as a
sanity check rather than a headline quality result.

The three-shard Qwen2.5-1.5B HellaSwag evaluation is now complete with no
underfilled runs: each policy has 768 paired examples. BF16 normalized accuracy
is 63.67%, while K4V4 reaches 63.15%, a paired change of -0.52 points (95% CI:
-1.43 to +0.39). Equal-memory K8V4-minus-K4V8 is -0.26 points (CI: -1.30 to
+0.78), and K4V3-minus-K3V4 is -0.52 points (CI: -1.82 to +0.78). These
intervals do not resolve a policy ranking; the result supports ordinary-task
quality preservation at substantial cache compression rather than an
objective-specific allocation claim.

The corrected ARC-Challenge suite is also complete: three explicitly disjoint
99-example shards provide 297 paired examples per policy. BF16 raw accuracy is
45.12%, while K4V4 reaches 45.79%, a paired change of +0.67 points (95% CI:
-1.01 to +2.36). Equal-memory K8V4-minus-K4V8 is +0.34 points (CI: -0.67 to
+1.68), and K4V3-minus-K3V4 is -0.34 points (CI: -2.02 to +1.35). As with
HellaSwag, every interval includes zero. We therefore interpret the task suite
as evidence that K4V4 preserves ordinary downstream behavior at substantial
standalone-cache compression, not as evidence that quantization improves task
accuracy or that ARC resolves the K/V allocation question. Cross-family task
runs remain in flight.

The first corrected cross-family task aggregate is now complete for
Llama-3.2-3B, with 256 paired examples per task. On ARC-Challenge, BF16 and K3V4
both score 42.58%; K4V4 changes accuracy by +0.39 points (95% CI: -0.78 to
+1.95) while saving 53.88% of standalone cache bytes. On HellaSwag, BF16 scores
75.00% and K4V4 scores 76.95%, with a paired change of +1.95 points (CI: +0.39
to +3.91) and 58.57% cache savings. We treat the positive HellaSwag delta as
finite-sample preservation rather than a quantization improvement. All four
matched K/V comparisons span zero or touch zero: the task aggregate does not
resolve whether keys or values deserve more bits. OLMo-2 and SmolLM2 task runs
remain in flight.

The full joint target/draft grid is now complete: 25 precision combinations at
1K and 4K, with three disjoint seeds per context. Under target KL <= 0.01,
target top-1 >= 95%, runtime-fidelity limits, and an acceptance lower-confidence
bound of -2 points, the 1K selector chooses target K4V8 plus draft K8V4. It saves
54.59% of total target-plus-draft KV, changes acceptance by -0.46 points (95% CI:
-1.78 to +0.84), and has target KL 0.00177 and 97.69% target top-1 agreement.
This is approximate: five of 288 prompt occurrences are non-tie/unknown target
disagreements, compared with two in the BF16 baseline. The distribution-preserving
alternative keeps the target in BF16 and uses draft K4V4, saving 29.02% total KV
at 1K and 30.58% at 4K with acceptance CIs of -0.90 to +1.25 and -1.13 to +1.89
points. No target-quantized candidate passes the acceptance-confidence constraint
at 4K, so the constrained selector also chooses the exact-target K4V4 policy.

The powered verifier audit is complete on the same 32 prompts under BF16/FP32
and SDPA/eager attention, with 512 speculative decisions per condition. BF16
SDPA has six top-1 disagreements: one satisfies the predeclared `1e-3` target-
margin tie rule and five are non-tie/unknown. BF16 eager has seven disagreements:
four ties and three non-tie/unknown. Both FP32 SDPA and FP32 eager have zero.
Earlier focused tests found no causal-mask, causal-suffix, or cache-rollback
failure. We therefore attribute the remaining BF16 differences to finite-
precision kernel-path drift, but do not claim bitwise identity to tokenwise BF16
greedy decoding.

## Is This Sufficiently Novel For A Main Track?

KV-cache quantization alone is not novel. KIVI established asymmetric K/V
quantization, KVmix and RateQuant study importance-aware or rate-distortion bit
allocation, and QuantSpec and Quasar combine quantization with speculative
decoding. The potential main-track contribution is narrower:

> KV-cache sensitivity is jointly determined by tensor role, quantizer geometry,
> and downstream decoding objective; conclusions drawn from isotropic noise or
> one quantization axis do not reliably transfer to a deployed quantizer.

The current cross-family K4V4 result is strong evidence that draft-cache
quantization is practical, but it is not yet enough by itself for a main-track
novelty claim. The paper becomes substantially stronger if the powered study
establishes at least one of the following:

1. A statistically resolved, memory-matched allocation difference between
   speculative acceptance and ordinary quality.
2. An objective-aware allocator that beats uniform and quality-optimized
   baselines on held-out acceptance at the same memory budget.
3. A packed-kernel implementation demonstrating end-to-end throughput or batch
   capacity gains at long context, not only fake-quantization memory estimates.

If those do not hold, the honest contribution is still useful but should be
framed as a broad empirical finding: quantizer geometry reverses the apparent
K/V asymmetry, correct geometry largely aligns the two objectives, K4V4 is a
robust draft-cache operating point, and Gaussian noise can give misleading K/V
conclusions. The new quality-surrogate result adds a practical contribution: a
two-stage calibration procedure can use cheap ordinary-LM metrics as a
conservative gate and reserve expensive speculative evaluation for candidates
near the memory--quality frontier. This strengthens the systems methodology but,
without a new allocator or packed-kernel gain, is not yet sufficient on its own
for a main-track novelty claim.

## Why Not Quantize Ordinary LLM Caches Too?

We can and should. That is why every major configuration is evaluated under a
teacher-forced ordinary-LM objective as well as speculative decoding. The key
difference is correctness:

- Quantizing only the draft cache changes proposal quality and therefore
  acceptance and speed, but an exact full-precision target verifier preserves
  the final target distribution.
- Quantizing a standalone model's cache changes the model's output distribution,
  so it must be evaluated using KL/NLL, task accuracy, retrieval, and generation
  quality.
- Quantizing the speculative target/verifier cache also changes the final output
  distribution. It may save substantially more total memory, but the method is
  approximate rather than lossless and needs stricter quality and numerical
  exactness audits.

This gives us a clean comparative study: use the same quantization policy for
ordinary decoding, draft-only speculative decoding, and joint target/draft
quantization, then measure both common robustness structure and any genuine
objective-specific differences instead of assuming they exist.

## Experiments In Flight

- The powered disjoint-shard FineWeb-Edu Qwen2.5-7B/3B
  K4V3-versus-K3V4 test is complete.
- Qwen HellaSwag and eight-shot ARC-Challenge are complete. The first corrected
  Llama task aggregate is complete, while OLMo-2 and SmolLM2 are running or
  dependency-gated. A powered replication will cover 576 disjoint HellaSwag
  examples and 297 disjoint ARC examples per model.
- Controlled 1K/4K quantizer-geometry factorial replication, followed by a
  three-shard FineWeb-Edu matched grid over native, K8V4, K4V8, K4V4, K3V4,
  and K4V3.
- All-layer FineWeb-Edu calibration of separate ordinary-quality and
  speculative-acceptance allocations, followed by equal-budget cross-objective
  evaluation on disjoint held-out blocks. The Qwen profile jobs are running;
  OLMo-2 and Llama replications are dependency-gated behind them.
- A dependency-chained 4/8/16-sample calibration ablation will reconstruct
  sensitivity maps from the same per-example FineWeb rows and compare both
  risk rankings and the actual selected K/V bit maps without additional GPU
  inference.
- Long-context objective replication on OLMo-2 and Llama after their all-layer
  campaigns complete.
- Matched, explicitly disjoint C4, GSM8K, and HumanEval robustness evaluation
  with three seeds and nine precision policies, including aggressive 2-bit
  controls, followed by aggregation and paper integration.

## Closest Work

- [KIVI](https://arxiv.org/abs/2402.02750)
- [AsymKV](https://arxiv.org/abs/2410.13212)
- [Quantize What Counts / KV-AdaQuant](https://arxiv.org/abs/2502.15075)
- [QuantSpec](https://arxiv.org/abs/2502.10424)
- [KVmix](https://arxiv.org/abs/2506.08018)
- [RateQuant](https://arxiv.org/abs/2605.06675)
- [KVarN](https://arxiv.org/abs/2606.03458)
- [Quasar](https://arxiv.org/abs/2603.01399)
- [Cross-Model KV Cache Transfer](https://arxiv.org/abs/2608.03893)
