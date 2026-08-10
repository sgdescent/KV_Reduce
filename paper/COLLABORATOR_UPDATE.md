# KV-Cache Quantization: Collaborator Update

Status: provisional results as of August 10, 2026. Powered allocation,
long-context, task, and speculation-length experiments are still running. The
four-condition verifier exactness audit is complete.

## Copy-Paste Message

The cache-quantization pivot is promising, but I would not yet call the current
result sufficient for a main-track paper. KV-cache quantization, asymmetric K/V
precision, and mixed-precision search already have strong prior work. Our sharper
potential contribution is to show that the *downstream objective* matters: the
precision policy that preserves ordinary LM quality need not be the policy that
maximizes speculative acceptance and serving efficiency. A main-track case needs
a statistically resolved objective-specific allocation or an allocator that
beats ordinary-quality and uniform baselines at equal memory, ideally with packed
kernels and end-to-end long-context gains.

This should not be restricted to speculative decoding. We are now evaluating the
same quantizer under three deployment regimes: ordinary autoregressive decoding,
draft-only speculative quantization, and joint target/draft quantization. Ordinary
decoding is the control objective and a useful application in its own right. The
important distinction is that draft-only quantization remains distribution-exact
because the BF16 target verifier corrects every proposal, whereas quantizing a
standalone model or the target cache changes the output distribution and therefore
requires stricter KL, NLL, top-1, task-accuracy, and exactness constraints.

The strongest validated numbers so far are encouraging. Draft K4V4 saves 66.13%
of draft-cache storage and 22.64% of combined target-plus-draft KV with a +0.10
point macro acceptance change (95% CI: -0.04 to +0.25). Joint target K4V8 plus
draft K8V4 reaches 54.59% combined KV savings at 1K with a -0.46 point acceptance
change (CI: -1.78 to +0.84), but it is approximate and does not yet satisfy our
conservative acceptance bound at 4K. The exact-target alternative keeps target
BF16 and quantizes draft K4V4, saving 29.02% at 1K and 30.58% at 4K while retaining
the target distribution. The campaign is still testing longer contexts, tasks,
speculation lengths, and powered objective reversals.

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

The completed three-seed `gamma=2` test resolves this asymmetry at two bits.
Across 191 paired prompts, K4V2 loses 5.43 acceptance points from BF16 (95% CI:
-7.32 to -3.58), while K2V4 loses 1.43 points (CI: -2.58 to -0.31). The direct
paired K4V2-minus-K2V4 contrast is -4.01 points (CI: -5.82 to -2.15). The two
policies have the same nominal mean bit-width and near-equal estimated total-KV
savings (31.76% versus 31.38%); the small byte difference comes from asymmetric
key metadata. This is evidence that, with grouped per-channel KIVI keys,
aggressive value quantization can be more harmful than aggressive key
quantization. The `gamma=4` and `gamma=8` replications are still running.

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
quantization, then measure which objective selects which precision allocation.

## Experiments In Flight

- Powered Qwen2.5-7B/3B equal-memory K4V3 versus K3V4 test.
- C4, GSM8K, and HumanEval robustness evaluation.
- Eight-shot HellaSwag and ARC-Challenge task accuracy across disjoint seeds.
- 16K and 32K PG19 long-context evaluation.
- Speculation-length (`gamma`) sensitivity.

## Closest Work

- [KIVI](https://arxiv.org/abs/2402.02750)
- [QuantSpec](https://arxiv.org/abs/2502.10424)
- [KVmix](https://arxiv.org/abs/2506.08018)
- [RateQuant](https://arxiv.org/abs/2605.06675)
- [Quasar](https://arxiv.org/abs/2603.01399)
- [Cross-Model KV Cache Transfer](https://arxiv.org/abs/2608.03893)
