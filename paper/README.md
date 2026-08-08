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

## Current evidence status

- Qwen2.5-3B/1.5B 1K and 4K quantization results are complete.
- Gaussian K/V perturbation is complete.
- One Qwen top-eight-layer allocation is complete.
- Cross-family results are running under `outputs/iclr_spec_kv/` on Catalyst.
- Kernel-backed latency, downstream long-context quality, and multi-seed results
  are still required before submission.
