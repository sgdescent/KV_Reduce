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

The current objective-matrix aggregator accepts only `cached_dynamic_v4` speculative
summaries and `teacher_forced_cached_v1` ordinary-quality summaries, computes prompt-level
bootstrap confidence intervals and paired equal-memory effects, audits numerical
tie versus non-tie target mismatches, and writes paper-ready PDF/PNG figures plus
a LaTeX table.

## Current evidence status

- Earlier Qwen2.5-3B/1.5B results are retained only as preliminary evidence;
  final tables use the cache-resident evaluator campaign.
- Gaussian K/V perturbation is complete.
- A four-budget, three-context, three-seed Qwen objective matrix is running.
- Llama and OLMo cross-family profiles, an all-layer Qwen run, streaming C4,
  uncertainty-aware allocation, and a calibration-size ablation are dependency-chained.
- Equal-memory K-priority and V-priority heuristics are included in new matrices.
- Kernel-backed latency, downstream long-context quality, and multi-seed results
  are still required before submission.
