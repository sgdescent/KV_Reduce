# Memory-Efficient Speculative Decoding with Quantized Draft KV Cache

Saksham Gupta\*, Michael Li\* (Carnegie Mellon University) · AXIOM Workshop @ NeurIPS 2026 · \*equal contribution

[Paper](https://openreview.net/forum?id=NE193n19qO) · [Reproducing the experiments](docs/REPRODUCING.md) · [Citation](#citation)

Speculative decoding stores two KV caches, one for the target model and one for the draft model. We quantize only the **draft** cache. The target cache and verifier stay in BF16, so quantization can lower acceptance but cannot change the target model's output rule.

## Main results

- **Quantizer geometry decides the key/value split.** With a naive per-token quantizer, K8V4 beats equal-memory K4V8 by 10.87 acceptance points across six model pairs. With a KIVI-inspired layout (grouped per-channel keys, affine values and a BF16 residual window), the gap is −0.25 points.
- **K4V4 removes two thirds of the draft cache.** Across six target/draft pairs, draft-cache bytes drop by 66.20% and combined target + draft KV by 23.95%. Acceptance moves from 57.93% to 57.14%.
- **The storage savings are real, but not yet a speedup.** Bit-packed K4V4 stores a 32K-token Qwen2.5-1.5B cache in 260.2 MiB instead of 896.0 MiB. The end-to-end prototype is slower than BF16 speculative decoding and does not lower peak GPU memory.

## Setup

Run every command from the repository root. Python entry points are modules, e.g. `python -m kv_reduce.experiments.benchmark_spec_kv_quantization --help`.

**CPU check of Appendix E's released results** (no GPU or model downloads):

```bash
python -m pip install -r requirements-analysis.txt
python -m kv_reduce.analysis.analyze_axiom_review --check
```

To regenerate the Appendix E figures from the archived raw outputs (also CPU-only), see [results/axiom-review-2026](results/axiom-review-2026/README.md).

**GPU experiments** (Linux + CUDA; the paper used Python 3.13 and 24 GiB RTX A5000 GPUs):

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

Models and datasets download from Hugging Face; Llama checkpoints need gated access.

## Small example

Compare a BF16 and a K4V4 draft cache on two prompts with Qwen2.5-3B (target) and Qwen2.5-1.5B (draft):

```bash
python -m kv_reduce.experiments.benchmark_spec_kv_quantization \
  --big_model Qwen/Qwen2.5-3B --small_model Qwen/Qwen2.5-1.5B \
  --big_device cuda:0 --small_device cuda:0 --attn_implementation sdpa \
  --dataset_name HuggingFaceFW/fineweb-edu --dataset_config sample-10BT \
  --eval_split train --stream_eval --shuffle_eval --seed 0 \
  --prompt_len 1024 --num_prompts 2 --warmup_prompts 1 --skip_prompts 100 \
  --max_new_tokens 32 --draft_steps 4 --target_verification_mode sequential \
  --quant_configs 'none;k4v4' --key_quant_axis per_channel \
  --key_group_size 32 --key_residual_length 0 --value_quant_scheme affine \
  --out_dir outputs/example
```

This writes per-prompt rows (`benchmark_rows.csv`) and a summary of acceptance, draft–target agreement and KV-cache bytes (`summary.json`). The evaluator uses fake quantization: it measures acceptance and bytes, not packed-cache speed.

## Repository layout

| Path | Contents |
| --- | --- |
| `kv_reduce/` | Library: quantizers (`kv_cache_quantization.py`), bit-packed cache (`packed_kv_cache.py`), Triton decode attention (`triton_packed_kv_attention.py`), model/data helpers, statistics |
| `kv_reduce/experiments/` | GPU entry points: speculative decoding, LM quality, retrieval, packed-cache benchmarks |
| `kv_reduce/allocation/` | Build mixed-precision K/V policies from sensitivity profiles (CPU) |
| `kv_reduce/analysis/` | Aggregate runs into the paper's statistics (CPU) |
| `scripts/` | SLURM launchers, one folder per experiment ([how to use them](docs/REPRODUCING.md#launchers)) |
| `results/` | Frozen outputs for the introduction's noise test, §5.2, Table 3 and Appendices D–E; `axiom-review-2026/` also holds Appendix E's raw outputs and figures. Other results must be rerun |
| `tests/` | CPU unit tests (`pytest`) |

## Paper → code

Main-code paths are relative to `kv_reduce/`.

| Paper result | Launchers | Main code |
| --- | --- | --- |
| Introduction: Gaussian KV noise | run directly | `experiments/kv_perturbation_sweep.py` |
| Table 2, Fig. 3a: K4V4 across six pairs | `scripts/cross_family/` | `experiments/benchmark_spec_kv_quantization.py` |
| §5.2: draft KL and top-1 agreement | `scripts/cross_family/` | `experiments/profile_kv_quality_sensitivity.py` |
| Table 3, Fig. 2a: per-token vs. grouped keys | `scripts/cross_family/` | `analysis/compare_kv_quantizer_geometry.py` |
| Table 4, App. B: key-axis, group-size and residual controls | `scripts/key_geometry/` | `analysis/aggregate_value_precision_sweep.py`, `analysis/aggregate_paired_geometry_sweep.py` |
| Table 5, Fig. 2b: 16K passkey retrieval | `scripts/passkey/` | `experiments/eval_kv_quantized_multiple_choice.py` |
| App. C: LongBench passage retrieval | `scripts/longbench/` | `experiments/eval_kv_quantized_multiple_choice.py` |
| Table 6: PG-19 at 16K and 32K | `scripts/long_context/` | `experiments/benchmark_spec_kv_quantization.py` |
| App. D, Fig. 3b: packed storage and kernel | `scripts/packed/` | `experiments/benchmark_packed_kv_attention.py` |
| App. E: 256-token runs, end-to-end prototype | `scripts/axiom_review/` | `experiments/packed_review.py` |
| App. F: choosing a precision policy | `scripts/precision_policy/` | `allocation/search_kv_bit_allocation.py` |

[docs/REPRODUCING.md](docs/REPRODUCING.md) gives the commands and aggregation steps for each row, and names the frozen outputs where they exist.

## Limitations

The packed decoder is a research prototype. It is slower than BF16 speculative decoding and does not reduce peak GPU memory. Its batched verifier has unresolved output mismatches, also with a BF16 draft cache; all sequential checks pass. Tables 3–4 and the Appendix F Spearman grid also used batched verification (paper, Appendix A). Acceptance intervals that include zero do not establish equivalence. See [correctness and scope](docs/AXIOM_RESULTS.md#correctness-and-scope).

## Citation

```bibtex
@misc{gupta2026quantizeddraftkv,
  title  = {Memory-Efficient Speculative Decoding with Quantized Draft KV Cache},
  author = {Gupta, Saksham and Li, Michael},
  year   = {2026},
  note   = {Accepted at the AXIOM workshop at NeurIPS 2026},
  url    = {https://openreview.net/forum?id=NE193n19qO}
}
```

## License

Code is released under the [MIT License](LICENSE). Models and datasets keep their upstream terms (paper, Table 10). No model weights are redistributed here; the raw outputs in `results/axiom-review-2026/` include the token IDs of their FineWeb-Edu prompts. [Contributions](CONTRIBUTING.md) are welcome.
