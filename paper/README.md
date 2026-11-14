# AXIOM 2026 Short Paper

This directory contains the camera-ready AXIOM paper (short-paper track):

**Memory-Efficient Speculative Decoding with Quantized Draft KV Cache**

It uses the official NeurIPS 2026 `dblblindworkshop` style with the `final`
option, which shows the authors. The
paper focuses on reducing the extra KV-cache memory required by two-model
speculative decoding. The quantizer-layout study explains how to choose key and
value precision without drawing misleading conclusions.

Build from this directory:

```bash
latexmk -pdf -interaction=nonstopmode main.tex
```

`figures/geometry_interaction.py` and `figures/memory_acceptance_landscape.py`
regenerate Figures 2 and 3 from the repository root.

Camera-ready rules from the AXIOM call for papers:

- The main paper is at most four pages (pages 1 to 4). References and the
  appendix do not count, and the camera-ready appendix may be any length.
- Keep `\workshoptitle{AXIOM Foundations of Efficient Deep Learning}` as the
  call for papers writes it, without a colon.

The current build is twelve pages: four content pages, two reference pages,
and six appendix pages. Page 4 has about one line free, so any addition to the
main text needs an offsetting cut.
