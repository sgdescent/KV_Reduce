# AXIOM 2026 Short Paper

This directory contains the separate four-page, double-blind AXIOM submission:

**Memory-Efficient Speculative Decoding with Quantized Draft KV Caches**

It uses the official NeurIPS 2026 `dblblindworkshop` style linked by AXIOM. The
paper focuses on reducing the extra KV-cache memory required by two-model
speculative decoding. The quantizer-layout study explains how to choose key and
value precision without drawing misleading conclusions.

Build from this directory:

```bash
latexmk -pdf -interaction=nonstopmode main.tex
```

Submission layout used by this draft:

- The main paper occupies pages 1 to 4.
- References begin on page 5.
- The supplementary appendix starts after the references.
- The appendix must not exceed six pages.

Check the page-4 content boundary, the page-5 reference boundary, and the
six-page appendix maximum after every substantive edit. The current build is
eight pages: four content pages, one reference page, and three appendix pages.
