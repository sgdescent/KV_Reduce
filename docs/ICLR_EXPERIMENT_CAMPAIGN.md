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

The launcher uses matching array dependencies, so a failed smoke task prevents
only that model pair from advancing. Sustained stages are capped at two GPUs.
Jobs exclude Catalyst nodes `catalyst-0-9` (flaky GPU/prolog behavior observed)
and `catalyst-0-15` (down at campaign launch).

## Metrics

- speculative token acceptance rate and accepted tokens per verification round;
- target/draft top-1 match, Jensen-Shannon divergence, and acceptance mass;
- draft and total KV-cache bytes saved;
- exact-match against greedy target decoding;
- measured runtime only as a diagnostic, because the current fake-quantization
  path dequantizes in Python and is not a kernel-level speed benchmark.

## Launch

```bash
bash scripts/submit_iclr_spec_kv_campaign.sh
```

To run only a subset of pairs:

```bash
ARRAY_RANGE=0-2 CAMPAIGN_CONCURRENCY=2 \
  bash scripts/submit_iclr_spec_kv_campaign.sh
```
