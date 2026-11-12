# Reproduce the AXIOM review experiments (Appendix E)

This guide covers the **October 19, 2026 review campaign**: 256-token continuations
and the end-to-end packed prototype. [REPRODUCING.md](REPRODUCING.md) covers the
other paper experiments. No script in this guide changes the manuscript.

## CPU analysis

From the repository root:

```bash
python -m pip install -r requirements-analysis.txt
python -m kv_reduce.analysis.analyze_axiom_review --check
python -m unittest discover -s tests -p 'test_axiom_review_release.py' -v
```

The frozen bundle contains 576 quality configuration–prompt rows, 108 scored
batched systems rows plus 6 warmup rows, 94 sequential audits (including warmups),
and 38 target-only rows (including 2 warmups). Analysis excludes warmups from
quality/performance means. Audits occur on batched mismatches; they are not an
independent random sample. All audit outcomes are reported, including warmups.

The checker validates counts, pairing, token hashes, round totals and recorded
correctness outcomes, then compares reproduced summary values with the frozen
reference. It cannot inspect omitted token sequences or verify GPU computations.

## GPU environment

Use Linux with an NVIDIA CUDA GPU. The recorded packed runs used one **RTX A5000
with 24 GiB**. Target and draft live on the same GPU. CPU analysis needs no CUDA.
The packed prototype imports Triton and Transformers' Qwen2 internals; arbitrary
library upgrades are not a supported compatibility claim.

Recorded quality-run versions were PyTorch **2.10.0+cu128** and Transformers
**5.3.0**. `requirements.txt` also records packages read from the environment
on November 10. Those later observations are not a complete historical lockfile.
Use Python 3.13 to match the documented campaign environment when practical.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

Select a driver compatible with that CUDA wheel. Other GPUs, drivers and package
versions can change numerical behavior and timing.

Models and FineWeb-Edu are downloaded from their upstream sources. Follow their
access and usage terms. The original campaign did not pin model or dataset revision
hashes; fresh downloads therefore do not promise bit-for-bit historical replication.
Record revisions, software, hardware and launch arguments for new runs.

## Small quality example

The [README example](../README.md#small-example) is a two-prompt smoke run of the
quality evaluator. It is not a reproduction of the reported scientific result. The
quality evaluator uses **fake quantization**; its timing and memory estimates must
not be presented as packed-cache serving performance.

For the six full quality settings and two packed benchmarks, see the
[quality matrix](#quality-matrix). Launchers refuse to overwrite existing run directories.


## Quality matrix

Each task evaluates 24 scored prompts plus 1 warmup, four configurations (BF16,
K8V4, K4V8, K4V4), proposal length 4 and 256 generated tokens. Prompt selection uses
FineWeb-Edu `sample-10BT`, streamed `train`, seeded shuffle and 100 skipped blocks.
Key groups contain 32 **tokens** per channel; values use per-token affine quantization.

| Task | Context tokens | Residual key tokens | Prompt seed |
| --- | ---: | ---: | ---: |
| 0 | 1,024 | 0 | 0 |
| 1 | 1,024 | 128 | 0 |
| 2 | 1,024 | 0 | 1 |
| 3 | 1,024 | 128 | 1 |
| 4 | 16,384 | 0 | 0 |
| 5 | 16,384 | 128 | 0 |

```bash
bash scripts/axiom_review/quality.sh 0 --dry-run
bash scripts/axiom_review/quality.sh 0
```

Run the other task IDs separately when you intend to reproduce the full matrix.
`PYTHON` chooses the interpreter; `OUT_ROOT` chooses the output parent. Relative
output paths resolve from the repository root. Existing output directories are
rejected, including incomplete runs, so a retry must use a new output parent.

The quality evaluator uses sequential verification and fake quantization. Its
latency is not a packed serving benchmark, and its target calls do not demonstrate
acceleration. The campaign evaluator `review_spec.py` is byte-identical to
`benchmark_spec_kv_quantization.py` at commit `f908626`; the repository
reorganization later moved it to `kv_reduce/experiments/` and changed only its
import lines. To check the hash recorded in `provenance.json`:

```bash
git show f908626:benchmark_spec_kv_quantization.py | sha256sum  # 95b9f6ac...
```

## Packed prototype matrix

| Task | Context tokens | Scored prompts | Generated tokens |
| --- | ---: | ---: | ---: |
| 0 | 1,024 | 24 | 256 |
| 1 | 16,384 | 12 | 256 |

```bash
bash scripts/axiom_review/packed.sh 0 --dry-run
# Small CUDA smoke run: 1 scored prompt, 1 warmup, 32 generated tokens.
python -m kv_reduce.experiments.packed_review --smoke --out outputs/axiom-packed-smoke
# Full task; choose a fresh output directory.
bash scripts/axiom_review/packed.sh 0
```

The script runs append/crop/commit and attention checks before model evaluation.
It compares target-only, BF16 speculative, packed residual 0 and packed residual
128. Prompt selection uses seed 73 and 200 skipped blocks, unlike the quality
matrix. Do not pair results between these two prompt populations.

The target cache stays BF16. The packed draft uses affine quantization with BF16
metadata. Decode reads packed tensors through Triton; prefill first materializes
BF16 KV. Target-only memory is measured before loading the draft. Timing includes
prefill but excludes separate sequential audits; warmups are excluded in analysis.
Configuration order alternates across prompts. Runs were on a shared node.

**Known issue:** batched sequences frequently differ from target-only greedy
sequences even with BF16 draft KV. A mismatch triggers a sequential audit; a failed
sequential audit aborts the run. Do not interpret a completed job as certification
of batched output equivalence. The release adds an explicit warning, validates
positive arguments and makes `--smoke` functional; the decoding algorithm is
unchanged from the archived campaign prototype.

## Optional Slurm submission

These commands submit GPU work; run them only when ready. Configure your cluster's
partition/account and activate your GPU environment before submission.

```bash
mkdir -p logs
sbatch --array=0-5%2 scripts/axiom_review/review.slurm quality
sbatch --array=0-1%1 scripts/axiom_review/review.slurm packed
```

The template requests 1 GPU, 4 CPU cores, 48 GiB host memory and up to 12 hours per
task. It has no hard-coded account, private home directory or excluded node names.
No automated retry or matrix expansion is included.

## Metrics and provenance

Read [AXIOM_RESULTS.md](AXIOM_RESULTS.md) for the estimands and claim boundaries.
The released [provenance file](../results/axiom-review-2026/provenance.json) records
source hashes and distinguishes historical runtime metadata from later environment
observations. Original raw outputs remain preserved by the authors. Published
metrics omit source text and token sequences; token hashes preserve equality checks.

The frozen bundle and `--check` are for this campaign. New GPU runs produce the
original raw schemas, including token arrays. Review and sanitize new outputs before
publishing them; do not copy complete private campaign manifests into the repository.
