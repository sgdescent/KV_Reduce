# ICLR Experiment Campaign

This campaign tests whether speculative acceptance has a consistent asymmetric
dependence on draft-cache key and value precision across model families and
model-size gaps.

## Primary hypothesis

At equal cache memory, preserving key precision and reducing value precision
(`K8V4`) should retain more speculative acceptance than the reverse allocation
(`K4V8`). Keys determine attention routing through `softmax(QK^T)`, while values
carry the payload after routing. Final outputs remain exact because the target
verifier is always full precision.

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
5. `robustness`: two shuffled C4 validation seeds for representative Qwen,
   Llama, and OLMo pairs.
6. `long_context`: exploratory 8K and 16K Qwen2.5 runs before increasing the
   prompt count for the final long-context result.

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
- draft and total KV-cache bytes saved;
- exact-match against greedy target decoding;
- measured runtime only as a diagnostic, because the current fake-quantization
  path dequantizes in Python and is not a kernel-level speed benchmark.

The evaluator performs one target and one draft prefill per prompt, reuses both
dynamic caches across speculative rounds, verifies each proposal against the
existing target cache, and crops rejected suffixes in place. Target-greedy
reference generation is performed once outside each timed configuration. Runs
missing `runtime.evaluator_version=cached_dynamic_v3` are rejected by the paper
aggregation script.

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

To run only a subset of pairs:

```bash
ARRAY_RANGE=0-2 CAMPAIGN_CONCURRENCY=2 \
  bash scripts/submit_iclr_spec_kv_campaign.sh
```
