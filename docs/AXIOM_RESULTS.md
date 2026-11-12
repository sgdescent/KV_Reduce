# AXIOM review results

These results come from the October 2026 campaign. They supplement, rather than replace, the original paper experiments. All eight AXIOM tasks completed. The November release reanalyzes preserved outputs; it does not contain new GPU runs.

## Quality over 256-token continuations

Qwen2.5-3B target / Qwen2.5-1.5B draft; 24 scored prompts per setting; one excluded warmup. BF16 target cache, greedy sequential verification, proposals up to four tokens. Grouped per-channel keys (32 tokens/group), affine per-token values. All configurations use identical prompts within each setting.

| Context / residual / seed | BF16 acceptance % | K4V4 acceptance % | K4V4 − BF16, pp [95% CI] |
| --- | ---: | ---: | --- |
| 1024 / 0 / 0 | 76.69 | 76.61 | -0.08 [-1.18, 0.94] |
| 1024 / 0 / 1 | 74.01 | 73.45 | -0.56 [-1.30, 0.17] |
| 1024 / 128 / 0 | 76.69 | 77.02 | +0.33 [-0.43, 1.06] |
| 1024 / 128 / 1 | 74.01 | 73.42 | -0.59 [-1.34, 0.20] |
| 16384 / 0 / 0 | 79.62 | 79.20 | -0.42 [-1.42, 0.52] |
| 16384 / 128 / 0 | 79.62 | 79.55 | -0.07 [-1.04, 0.95] |

Acceptance is the mean of per-prompt accepted/proposed fractions. This is different from a pooled ratio of token totals. Differences are **percentage points**, not relative percentages. All intervals include zero; this is not proof of equivalence, no degradation, or noninferiority. The 16K setting has one prompt-selection seed; 1K has two. Do not treat repeated residual configurations as independent prompts.

The released summary also contains K8V4 and K4V8. This grouped-only campaign does not repeat the paper’s controlled key-geometry intervention or its six-model study.

## Residual-window comparison

| Context / seed | K4V4 residual 128 − residual 0, pp [95% CI] |
| --- | --- |
| 1024 / 0 | +0.408 [-0.379, 1.290] |
| 1024 / 1 | -0.033 [-0.614, 0.629] |
| 16384 / 0 | +0.351 [-0.358, 1.110] |

No tested setting establishes an acceptance advantage for 128 residual tokens. Residual 0 saves more persistent cache storage. This supports a local tradeoff, not a universal optimal residual length.

## Packed systems measurements

| Context / scored n | Configuration | Mean seconds | Draft bytes saved % | Peak allocated GiB | Batched exact sequences |
| --- | --- | ---: | ---: | ---: | ---: |
| 1024 / 24 | Target only | 7.10 | — | 5.98 | Reference |
| 1024 / 24 | BF16 speculative | 9.68 | 0.00 | 8.87 | 3/24 |
| 1024 / 24 | Packed K4V4 r0 | 12.77 | 70.45 | 8.87 | 4/24 |
| 1024 / 24 | Packed K4V4 r128 | 12.87 | 67.05 | 8.87 | 5/24 |
| 16384 / 12 | Target only | 8.65 | — | 7.69 | Reference |
| 16384 / 12 | BF16 speculative | 14.54 | 0.00 | 10.76 | 2/12 |
| 16384 / 12 | Packed K4V4 r0 | 17.20 | 71.05 | 10.76 | 2/12 |
| 16384 / 12 | Packed K4V4 r128 | 17.46 | 70.78 | 10.76 | 2/12 |

At 1K, the paired latency increases for residual 0/128 are +3.09 s [2.77, 3.42] / +3.18 s [2.73, 3.77]. At 16K, they are +2.66 s [2.30, 3.03] / +2.92 s [2.55, 3.22]. The packed path is slower; BF16 speculative decoding is also slower than target-only in this workload.

Persistent draft bytes include payloads, scales, offsets and residual tensors. They exclude target KV and model weights. Peak allocated memory is unchanged between speculative variants because storage reduction does not eliminate the BF16 prefill allocation. Peak allocation, allocator reservation and persistent KV storage are different quantities; reserved-byte measurements are included in the released per-prompt data.

Timings include prefill, exclude warmups and extra sequential audits, and were measured on a shared node. Target-only memory was measured before loading the draft. These are fixed-length prototype costs, **not validated exact-output serving comparisons**.

## Correctness and scope

- All 576 quality configuration–prompt evaluations matched target-only greedy tokens under sequential verification. Prompts repeat across configurations and residual variants.
- All 94 recorded packed sequential audits passed, including warmups. Audits were triggered by batched mismatches; they are not 94 independent randomly selected prompts.
- Batched sequence agreement is poor in both BF16 and packed configurations (table above). Quantization alone cannot explain BF16 mismatches. The root cause remains unresolved; do not presume numerical ties without diagnosis.
- Append/crop/commit and packed-attention smoke checks passed in the original GPU campaign. CPU release checks do not rerun these GPU checks.
- The campaign does not validate stochastic speculative sampling, production concurrency, maximum batch size, a lower OOM boundary or generalization across new model families.

## Prefixes, rejection positions and calls

| Context / residual / seed | BF16 prefix | K4V4 prefix | K4V4 − BF16, tokens [95% CI] |
| --- | ---: | ---: | --- |
| 1024 / 0 / 0 | 3.046 | 3.047 | +0.001 [-0.040, 0.041] |
| 1024 / 0 / 1 | 2.948 | 2.923 | -0.025 [-0.055, 0.003] |
| 1024 / 128 / 0 | 3.046 | 3.064 | +0.018 [-0.012, 0.047] |
| 1024 / 128 / 1 | 2.948 | 2.921 | -0.027 [-0.058, 0.005] |
| 16384 / 0 / 0 | 3.167 | 3.151 | -0.015 [-0.055, 0.024] |
| 16384 / 128 / 0 | 3.167 | 3.168 | +0.001 [-0.037, 0.043] |

Accepted-prefix length excludes correction/bonus tokens. Final rounds may propose fewer than four tokens. Per-prompt round distributions appear in `prefix-summary.json`; rejection categories are `[-1, 0, 1, 2, 3]`, where −1 means all proposed tokens were accepted and 0–3 are zero-based first rejected positions.

Every sequential quality run makes 257 target calls for 256 emitted tokens (one prefill plus tokenwise steps). Its 1.00390625 calls/token is **not a target-call saving**. Accepted prefixes alone do not establish acceleration. Calls can also differ in the number of tokens they process.

## Statistical procedure

Use 10,000 paired prompt-bootstrap resamples, conditional on each run’s setting. These exploratory intervals are not adjusted for multiple comparisons. Tokens and rounds within a prompt are not independent resampling units. The new campaign does not justify a cross-model population confidence interval.

Acceptance and packed statistics reproduce the frozen NumPy RNG seed 1910 analysis with the original configuration order. Residual contrasts use seed 1911. Prefix summaries use seed 1109, reset per task, with common resamples across configurations. The analysis checks all frozen numerical values. It does not silently choose a more favorable seed or omit mismatches.

## Remaining work

Diagnose the first batched mismatch in BF16 before changing the packed path: compare masks, cache positions, rollback state, target logits and top-two margins. Then validate any repair on a small correctness workload before timing it. Efficient packed kernels and lower-peak prefill are further engineering work, not results demonstrated here.

## Provenance and release scope

[Metrics and hashes](../results/axiom-review-2026/) · [Reproduction instructions](REPRODUCING_AXIOM.md) · [Paper](https://openreview.net/forum?id=NE193n19qO)

Campaign base commit: `41835f80a87c4dabc9b9040a619aecea79c4d71f`. The frozen summaries contain only AXIOM measurements. Private campaign manifests, Google Docs links and coauthor contact addresses are excluded. Compact metrics omit token arrays; the separately documented [raw ZIP](../results/axiom-review-2026/README.md) now preserves all 24 original review output files, including saved prompt and output/reference token IDs.
