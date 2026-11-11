# Quantized Draft KV Cache

**Memory-Efficient Speculative Decoding with Quantized Draft KV Cache**

Saksham Gupta · Michael Li

Accepted at **AXIOM, a NeurIPS 2026 workshop**

[Paper on OpenReview](https://openreview.net/forum?id=NE193n19qO) · [Reproduce the experiments](docs/REPRODUCING_AXIOM.md) · [Results and limitations](docs/AXIOM_RESULTS.md) · [Citation](CITATION.cff)

Compress the **draft model's KV cache** in two-model speculative decoding, while keeping the target cache in BF16. We study how key/value precision interacts with quantizer geometry, then measure acceptance and actual packed-cache storage.

> **Research prototype.** The October review experiments demonstrate smaller persistent draft-cache storage. They do **not** demonstrate a decoding speedup or lower peak GPU allocation. The packed benchmark's batched verifier has unresolved output mismatches, including with a BF16 draft. See [correctness and scope](docs/AXIOM_RESULTS.md#correctness-and-scope) before interpreting its timings.

## What this repository contains

- **Quantization:** grouped per-channel keys, per-token values, configurable bit widths and residual windows.
- **Evaluation:** paired acceptance measurements, sequential reference checks, long-context and cross-model research tools.
- **Packed storage:** actual packed payloads, metadata and residual tensors, with a Triton decode-attention prototype.
- **Review experiment release:** portable launchers, per-prompt metrics, bootstrap analysis and documented negative results.

The original research history is retained. The repository previously focused on translating KV caches between models; those instructions are preserved in [the legacy guide](docs/LEGACY_EXPERIMENTS.md). The `paper/` directory contains an **older research draft**, not the accepted AXIOM manuscript. Use the OpenReview link above for the paper.

## Results at a glance

The October 2026 review campaign used **Qwen2.5-3B / Qwen2.5-1.5B**, greedy decoding, batch size 1, proposals of up to 4 tokens, and **256-token continuations**.

| Measurement | Observed result | Interpretation |
| :--- | :--- | :--- |
| K4V4 acceptance vs. BF16 | −0.59 to +0.33 percentage points across six settings | Small central estimates; every paired 95% interval includes zero. Not an equivalence result. |
| Persistent packed draft-cache bytes | **67.05–71.05% lower** | Includes payload, metadata and residual storage; not total GPU memory. |
| Packed latency vs. BF16 speculative | **18.3–32.9% higher** | The prototype is slower. |
| Peak allocated GPU memory | **No reduction** | BF16 KV is materialized during prefill. |
| Sequential quality output agreement | **576/576** configuration–prompt evaluations | Repeated prompts across configurations; not 576 independent prompts. |
| Batched prototype output agreement | **3–5/24 at 1K; 2/12 at 16K** | Correctness remains unresolved; these are not validated exact-output serving timings. |

These new runs cover **one model pair**. They are separate from the original paper's six-model, shorter-generation study. Full tables, sample sizes, uncertainty definitions and residual-window comparisons are in [the results guide](docs/AXIOM_RESULTS.md).

## Start here: inspect results without a GPU

From the repository root, using Python 3.11 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-analysis.txt
python analyze_axiom_review.py --check
```

This validates the released metrics and reconstructs the acceptance, latency, storage, residual and prefix summaries. It downloads no models and submits no jobs. To save a fresh report:

```bash
python analyze_axiom_review.py --output outputs/axiom-review-summary.json
```

[Released metrics](results/axiom-review-2026/) include per-prompt and per-round statistics plus token-sequence hashes. Prompt text, generated token sequences, model weights and private cluster metadata are excluded. Hashes support consistency checks, not reconstruction of omitted outputs.

## Run a small quality experiment

GPU inference uses a separate environment. Follow [installation and hardware notes](docs/REPRODUCING_AXIOM.md#gpu-environment) first; the packed prototype requires Linux, CUDA and Triton.

```bash
python benchmark_spec_kv_quantization.py \
  --big_model Qwen/Qwen2.5-3B \
  --small_model Qwen/Qwen2.5-1.5B \
  --big_device cuda:0 --small_device cuda:0 \
  --attn_implementation sdpa \
  --dataset_name HuggingFaceFW/fineweb-edu --dataset_config sample-10BT \
  --eval_split train --stream_eval --shuffle_eval --seed 0 \
  --prompt_len 1024 --num_prompts 2 --warmup_prompts 1 --skip_prompts 100 \
  --max_new_tokens 32 --draft_steps 4 --target_verification_mode sequential \
  --quant_configs 'none;k4v4' --key_quant_axis per_channel \
  --key_group_size 32 --key_residual_length 0 --value_quant_scheme affine \
  --out_dir outputs/axiom-quality-smoke
```

This is a smoke workload, not a reproduction of the reported scientific result. The quality evaluator uses **fake quantization**; its timing and memory estimates must not be presented as packed-cache serving performance.

For the six full quality settings and two packed benchmarks, see the [reproduction guide](docs/REPRODUCING_AXIOM.md). Launchers refuse to overwrite existing run directories.

## Repository map

| Path | Purpose |
| :--- | :--- |
| `kv_cache_quantization.py` | Quantization layouts and byte estimates |
| `benchmark_spec_kv_quantization.py` | Quality evaluator, sequential/batched verifier modes and round records |
| `packed_kv_cache.py` | Bit packing, affine quantization and stored-byte accounting |
| `triton_packed_kv_attention.py` | Attention over packed KV tensors |
| `packed_review.py` | Qwen2 review benchmark; batched correctness caveat applies |
| `analyze_axiom_review.py` | CPU-only validation and paired bootstrap analysis |
| `scripts/axiom/` | Portable review experiment launchers |
| `results/axiom-review-2026/` | Released metrics, summaries and provenance |
| `tests/` | Quantizer, cache and analysis tests |
| `docs/` | Reproduction, results and historical research notes |
| `paper/` | Historical draft and figure-generation artifacts; not the accepted paper |

## Citation

Please cite the paper when using this work. This is a workshop acceptance, not a NeurIPS main-conference paper.

```bibtex
@misc{gupta2026quantizeddraftkv,
  title  = {Memory-Efficient Speculative Decoding with Quantized Draft KV Cache},
  author = {Gupta, Saksham and Li, Michael},
  year   = {2026},
  note   = {Accepted at the AXIOM workshop at NeurIPS 2026},
  url    = {https://openreview.net/forum?id=NE193n19qO}
}
```

## Contributing and license

For bug reports, include the command, software versions, hardware and a minimal reproducer. Keep correctness checks separate from performance comparisons. See [CONTRIBUTING.md](CONTRIBUTING.md).

Repository code is released under the [MIT License](LICENSE). Third-party models, datasets and bundled template files retain their own licenses and terms. No model weights or dataset passages are included in the new review-results bundle.
