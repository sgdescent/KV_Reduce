# AXIOM experiment outputs

This directory contains the **complete result-file archive for the October 2026
AXIOM review campaign**, plus derived metrics and regenerated figures.

## For Michael: regenerate the plots

From the repository root:

```bash
python -m pip install -r requirements-plots.txt
python plot_axiom_review.py
```

No GPU, Catalyst login, model weights or dataset download is needed. The command
verifies the ZIP and each original file, rebuilds metrics from the raw rows,
recomputes the paired-bootstrap summaries, checks them against the frozen results,
and writes five figures to `outputs/axiom-review-plots/`.

Each figure is exported as **PDF, editable SVG and PNG**, with a **CSV of the exact
plotted values**. Edit colors, labels or layout in `plot_axiom_review.py`, then rerun.
The plot script overwrites its figure output files; choose `--out YOUR_DIRECTORY`
to keep separate versions. A full `summary.json` is also written.

The generated files are already committed in [figures/](figures/):

| Figure | What it shows |
| --- | --- |
| [Acceptance](figures/acceptance_delta.pdf) | K4V4 vs. BF16 acceptance differences with paired intervals |
| [Residual window](figures/residual_delta.pdf) | Residual 128 vs. residual 0 |
| [Accepted prefixes](figures/accepted_prefix_delta.pdf) | Change in accepted draft prefix length |
| [Storage and runtime](figures/storage_runtime.pdf) | Persistent storage, prototype latency and peak allocation |
| [Rejection positions](figures/rejection_positions.pdf) | Where draft proposals first fail verification |

These are figures for the **new review campaign**. They do not replace or reproduce
all figures in the original six-model paper. Existing historical inputs and
figure builders remain under `paper/`; some older full experiment outputs have not
been added to this archive. The manuscript itself has not been edited.

## Original output files

[raw-outputs.zip](raw-outputs.zip) contains **24 files** from all nine result folders:

- Six `axiom_quality_*` folders: `benchmark_rows.csv` and `summary.json` in each.
- Two `axiom_packed_ctx*` folders: `complete.json`, `prompts.json`,
  `speculative.jsonl` and `target_only.jsonl` in each.
- One `axiom_packed_smoke` folder with the same four files, kept separately from
  the scored full runs. The plot analysis excludes this smoke workload.

The archive is about **1.2 MB compressed / 14.3 MB uncompressed**. Every file is
byte-identical to the corresponding Catalyst result file, verified on November 12,
2026. [raw-manifest.json](raw-manifest.json) records the archive checksum, each file's
SHA-256 and its uncompressed size. Original configuration paths are retained as
historical metadata, not instructions to use those directories.

Inspect or unpack with Python's standard library, into a new directory:

```bash
python -m zipfile -l results/axiom-review-2026/raw-outputs.zip
python -m zipfile -e results/axiom-review-2026/raw-outputs.zip outputs/axiom-review-raw
```

To verify checksums without generating plots:

```bash
python raw_axiom_archive.py
```

To reconstruct the compact metrics into an empty directory for custom analysis:

```bash
python raw_axiom_archive.py --out outputs/axiom-review-rebuilt
python analyze_axiom_review.py --data outputs/axiom-review-rebuilt --output outputs/axiom-review-rebuilt-summary.json
```

The reconstructed directory contains inputs, not the frozen reference files;
use the default bundled directory for `analyze_axiom_review.py --check`.

## Data and provenance

The compact top-level JSONL files omit token arrays. The ZIP now includes the
original saved output/reference token IDs and the packed runs' saved prompt IDs,
so token-level inspection is possible. Quality-run CSVs contain output/reference
IDs and round records; original quality prompt IDs were not saved by that evaluator.
Do not infer that the archive contains unsaved data, model checkpoints, scheduler
logs or every historical project experiment.

Prompts originate from [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu),
configuration `sample-10BT`; the dataset's upstream terms and attribution apply.
The token arrays can be decoded using the corresponding Qwen tokenizer. Model
weights are not included. No FMTS data, private shared-document links, contact
addresses or full campaign manifest are included.

Runs executed on **Catalyst**, using NVIDIA RTX A5000 GPUs. The campaign base code
was `41835f80a87c4dabc9b9040a619aecea79c4d71f`; the historical run environment and
code hashes are in [provenance.json](provenance.json). The released evaluator,
packed prototype and portable launchers are in the repository root and
`scripts/axiom/`.

## Interpretation

The acceptance and residual figures use 95% paired prompt-bootstrap intervals;
rounds and tokens are not independent replicates. The systems figure reports
prototype costs: batched output mismatches remain unresolved. See the
[full result definitions and limitations](../../docs/AXIOM_RESULTS.md).
