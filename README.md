# Quantized Draft KV Cache

**Memory-Efficient Speculative Decoding with Quantized Draft KV Cache**

Saksham Gupta · Michael Li — **AXIOM @ NeurIPS 2026**

[Paper](https://openreview.net/forum?id=NE193n19qO) · [Experiments](docs/REPRODUCING_AXIOM.md) · [Results](docs/AXIOM_RESULTS.md) · [Citation](CITATION.cff)

Speculative decoding keeps a KV cache for both a target model and a smaller draft model. We compress the **draft cache**, keeping the target cache in BF16.

## Main ideas

- **Quantizer geometry matters.** The choice of how to group keys and values changes which bit allocation works best.
- **Substantially smaller draft caches.** Our packed K4V4 implementation reduces persistent draft-cache storage by **67–71%** in the Qwen2.5 review experiments, including metadata and residual storage.
- **Small observed acceptance changes.** With 256-token continuations, K4V4 acceptance changes range from **−0.59 to +0.33 percentage points** versus BF16 across six settings for that model pair. Full results and paired confidence intervals are in the [results guide](docs/AXIOM_RESULTS.md).

The repository includes quantizers, packed-cache attention, experiment launchers, and released metrics with reproducible analysis.

## Quick start

Validate the released results on a CPU—no model downloads or GPU required:

```bash
python -m pip install -r requirements-analysis.txt
python analyze_axiom_review.py --check
```

To run model experiments, follow the [GPU setup and reproduction guide](docs/REPRODUCING_AXIOM.md).

**Regenerate plots without a GPU:** [download the complete review outputs and follow the plotting guide](results/axiom-review-2026/README.md).

## Explore

- [Reproduce experiments](docs/REPRODUCING_AXIOM.md): setup, small examples, full runs, and Slurm submission.
- [Read the results](docs/AXIOM_RESULTS.md): measurements, uncertainty, and interpretation.
- [Inspect released data](results/axiom-review-2026/): per-prompt metrics, summaries, and provenance.

## Cite this work

<details>
<summary>BibTeX</summary>

```bibtex
@misc{gupta2026quantizeddraftkv,
  title  = {Memory-Efficient Speculative Decoding with Quantized Draft KV Cache},
  author = {Gupta, Saksham and Li, Michael},
  year   = {2026},
  note   = {Accepted at the AXIOM workshop at NeurIPS 2026},
  url    = {https://openreview.net/forum?id=NE193n19qO}
}
```

</details>

Code: [MIT License](LICENSE). Third-party assets retain their own terms. [Contributions](CONTRIBUTING.md) are welcome.

## Current limitations

The packed decoder is a research prototype: it is currently slower than BF16 and does not reduce peak GPU allocation. Its batched verifier has unresolved output mismatches, including in the BF16 baseline; sequential checks passed. The acceptance intervals include zero, which does not establish equivalence. See [scope and correctness details](docs/AXIOM_RESULTS.md#correctness-and-scope).
