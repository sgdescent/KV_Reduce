# KV-Cache Quantization: Collaborator Update

Status: provisional results as of August 10, 2026. Powered replications,
long-context tests, and verifier exactness audits are still running.

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

Within the KIVI geometry, reducing value precision from four to three bits
(K4V3) is more harmful on average than reducing key precision (K3V4): K4V3
changes acceptance by -1.01 points and has KL 0.0327, while K3V4 changes
acceptance by -0.32 points and has KL 0.0111. Gaussian perturbation sensitivity
therefore cannot be treated as a direct proxy for quantization sensitivity.

We see one raw equal-memory objective-preference reversal on Qwen2.5-7B/3B:
speculative acceptance favors K4V3 over K3V4 by +0.43 points, while ordinary
quality significantly favors K3V4. However, the speculative confidence interval
is wide (-1.68 to +2.65 points), so this reversal is not statistically resolved.
A predeclared powered replication uses three new seeds, 512 speculative prompts
per seed, and 256 ordinary-quality sequences per seed.

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
quantization, then measure which objective selects which precision allocation.

## Experiments In Flight

- Powered Qwen2.5-7B/3B equal-memory K4V3 versus K3V4 test.
- Expanded cross-family target-cache quality and role comparison.
- C4, GSM8K, and HumanEval robustness evaluation.
- Eight-shot HellaSwag and ARC-Challenge task accuracy across disjoint seeds.
- Synthetic passkey retrieval at 4K, 8K, and 16K context and three insertion
  depths, using exact token-length construction.
- 16K and 32K PG19 long-context evaluation.
- Draft-only versus target-only versus joint target/draft quantization.
- Speculation-length (`gamma`) sensitivity.
- BF16/FP32 and SDPA/eager verifier exactness audit.

## Closest Work

- [KIVI](https://arxiv.org/abs/2402.02750)
- [QuantSpec](https://arxiv.org/abs/2502.10424)
- [KVmix](https://arxiv.org/abs/2506.08018)
- [RateQuant](https://arxiv.org/abs/2605.06675)
- [Quasar](https://arxiv.org/abs/2603.01399)
- [Cross-Model KV Cache Transfer](https://arxiv.org/abs/2608.03893)
