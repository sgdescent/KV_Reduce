# Citation Audit

Last verified: 2026-08-27

Every citation in `main.tex` was checked against an author-maintained archive
or an official proceedings record. The paper should not cite a remembered title,
an uncrosschecked search result, or metadata copied from a secondary summary.
Model reports with unusually large author lists use BibTeX's standard `and
others` form after a verified first author; their canonical records below retain
the complete lists.

| Cite key | Canonical source | Claim supported in this paper |
| --- | --- | --- |
| `leviathan2023speculative` | [PMLR 202](https://proceedings.mlr.press/v202/leviathan23a.html) | Speculative decoding evaluates several draft tokens in parallel while preserving the target output distribution. |
| `chen2023speculative` | [arXiv:2302.01318](https://arxiv.org/abs/2302.01318) | Speculative sampling uses a faster draft model and a target-preserving rejection-sampling correction. |
| `liu2024kivi` | [PMLR 235](https://proceedings.mlr.press/v235/liu24bz.html) | KIVI motivates per-channel key quantization and per-token value quantization. |
| `hariri2026quantize` | [ACL Anthology](https://aclanthology.org/2026.findings-acl.1314/) | Quantize What Counts argues for key-favored precision using key and value projection geometry. |
| `tiwari2025quantspec` | [PMLR 267](https://proceedings.mlr.press/v267/tiwari25b.html) | QuantSpec applies hierarchical quantized KV caches and quantized weights to self-speculative decoding. |
| `li2025kvtuner` | [PMLR 267](https://proceedings.mlr.press/v267/li25dd.html) | KVTuner searches for hardware-friendly layer-wise key and value precision pairs. |
| `zuo2026ratequant` | [arXiv:2605.06675](https://arxiv.org/abs/2605.06675) | RateQuant fits quantizer-specific distortion models and allocates precision across attention heads. |
| `liang2026blockgtq` | [arXiv:2606.24033](https://arxiv.org/abs/2606.24033) | Block-GTQ allocates key-cache precision across RoPE frequency blocks. |
| `yang2024qwen25` | [arXiv:2412.15115](https://arxiv.org/abs/2412.15115) | Identifies the Qwen2.5 model family used in the experiments. |
| `yang2025qwen3` | [arXiv:2505.09388](https://arxiv.org/abs/2505.09388) | Identifies the Qwen3 model family used in the experiments. |
| `grattafiori2024llama3` | [arXiv:2407.21783](https://arxiv.org/abs/2407.21783) | Identifies the Llama 3 model family used in the experiments. |
| `olmo2025furious` | [arXiv:2501.00656](https://arxiv.org/abs/2501.00656) | Identifies the OLMo 2 model family used in the experiments. |
| `allal2025smollm2` | [arXiv:2502.02737](https://arxiv.org/abs/2502.02737) | Identifies the SmolLM2 model family used in the experiments. |
| `merity2016pointer` | [arXiv:1609.07843](https://arxiv.org/abs/1609.07843) | Establishes the source of the WikiText corpus used for held-out prompts. |
| `bai2024longbench` | [ACL Anthology](https://aclanthology.org/2024.acl-long.172/) | Establishes the source and scope of the LongBench passage-retrieval evaluation. |
| `rae2020compressive` | [OpenReview](https://openreview.net/forum?id=SylKikSYDH) | Establishes the source of the PG-19 long-context corpus. |

## Corrections Made During This Audit

- Replaced outdated metadata for `arXiv:2502.15075` with the official ACL 2026
  Findings title, author list, pages, URL, and DOI.
- Replaced the KVTuner arXiv-style entry with its official ICML 2025 record and
  corrected the author order and page range.
- Expanded the QuantSpec author metadata and added the official PMLR volume.
- Rephrased Related Work so KVTuner, RateQuant, and Block-GTQ are not presented
  as performing the same kind of bit allocation.
- Described the implemented quantizer as KIVI-inspired and separated KIVI's
  published geometry from our chosen group sizes and residual window.
- Added primary-source citations for every named model family and public
  evaluation corpus or benchmark in the experimental section.

## Maintenance Rule

Before adding or changing a citation:

1. Open the official proceedings, anthology, DOI, or arXiv page.
2. Match title, author order, year, venue, pages, DOI, and URL.
3. Record the exact supported claim in this file.
4. Rebuild the paper and inspect the rendered reference entry.
