# Paper Draft

This directory contains an anonymous ICLR-style draft and reproducible preliminary
figures. The official ICLR 2027 author guide is live, but its linked `iclr2027.zip`
was not yet present in the official template repository when this draft was
created. The paper therefore uses the official ICLR 2026 style as a temporary
shell. Replace the style before submission.

## Build artifacts

```bash
python build_artifacts.py
latexmk -pdf -interaction=nonstopmode main.tex
```

Run these commands from `paper/`. The source JSON under `data/preliminary/` is
copied from completed Catalyst runs; figures and generated LaTeX tables can be
regenerated without editing paper numbers by hand.

For the cache-resident cross-family campaign, run from the repository root:

```bash
python paper/aggregate_campaign.py \
  --results_root outputs/iclr_spec_kv \
  --out_dir paper/campaign_artifacts
```

For the objective-aware KV quantization campaign, validated complete matrices can
be collected into common CSV, figure, and LaTeX artifacts with:

```bash
python paper/aggregate_objective_campaign.py \
  --results_root outputs/objective_kv \
  --out_dir paper/objective_campaign_artifacts

python paper/build_packed_artifacts.py

python paper/build_free_generation_artifacts.py \
  --source Qwen2.5-1.5B=outputs/kivi_free_generation_v1_retry/cross_family/qwen25_15b/aggregate/summary.json \
  --source Llama-3.2-3B=outputs/kivi_free_generation_v1_retry/cross_family/llama32_3b/aggregate/summary.json \
  --source OLMo-2-1B=outputs/kivi_free_generation_v1_retry/cross_family/olmo2_1b/aggregate/summary.json \
  --source SmolLM2-360M=outputs/kivi_free_generation_v1_retry/cross_family/smollm2_360m/aggregate/summary.json
```

By default, incomplete matrices, stale evaluator outputs, and smoke runs are not
included in the paper artifacts.

The current objective-matrix aggregator accepts only
`cached_dynamic_v6_sequential_target` speculative summaries and
`teacher_forced_cached_v1` ordinary-quality summaries. It requires sequential
BF16 target verification with an unquantized target cache, computes prompt-level
bootstrap confidence intervals and paired equal-memory effects, audits exact
target-greedy agreement, and writes paper-ready PDF/PNG figures plus a LaTeX
table. Earlier `cached_dynamic_v4` matrices are exploratory artifacts and are
not admitted to paper tables.

## Current evidence status

- The cache-resident speculative campaign is complete across six target/draft
  pairs. Draft-only results retain an unquantized BF16 verifier and are gated on
  complete paired prompts, supported evaluator versions, and target exactness.
- Matched ordinary-quality and speculative-acceptance objective matrices are
  complete. They show no resolved objective-specific allocation advantage.
- The powered 16K retrieval study contains 192 paired Qwen2.5 examples and finds
  a resolved key-axis-by-allocation interaction. The three-model LongBench
  passage-retrieval control is complete but saturates for Llama and Qwen3.
- The held-out retrieval-aware layer policy does not beat uniform policies and
  is reported as a null result.
- Actual packed storage is validated from 1K--32K. The direct Triton path avoids
  full-cache materialization but remains slower than native BF16 SDPA, so the
  draft makes no production throughput claim.
