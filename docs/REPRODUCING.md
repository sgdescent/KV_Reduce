# Reproducing the experiments

Each paper result has SLURM launchers under `scripts/` and a CPU aggregation step. Raw run outputs are not in the repository. Frozen data behind the paper's numbers is in `results/`.

Install the GPU environment first (see the [README](../README.md#setup)). All paths below are relative to the repository root.

## Launchers

- Submit from the repository root after `mkdir -p logs`.
  - `sbatch [--array=...] scripts/<experiment>/<name>.slurm` runs one job or array. Array launchers decode `SLURM_ARRAY_TASK_ID` into model pair, seed, context, etc. near the top of the file.
  - `bash scripts/<experiment>/submit_<name>.sh` submits a whole chain (runs, then aggregation) with SLURM dependencies.
- Settings are environment variables with defaults in each file (e.g. `OUT_ROOT`, `NUM_PROMPTS`). A chain can wait for another job through `AFTER_JOB`, `DEPENDENCY` or `START_DEPENDENCY` (named in the script). Most submit scripts honor `EXCLUDE_NODES=node1,node2`.
- Cluster options are not hard-coded. Set `SBATCH_PARTITION`/`SBATCH_ACCOUNT` or pass them to `sbatch`.
- Launchers activate `.venv/` if it exists. Outputs go to `outputs/` (git-ignored).
- Most launchers log to Weights & Biases. Run `wandb login`, or `export WANDB_MODE=offline`.
- Without SLURM, run the Python command inside a launcher directly.

## Model pairs

| Family | Target | Draft |
| --- | --- | --- |
| Qwen2.5 | `Qwen/Qwen2.5-3B` | `Qwen/Qwen2.5-1.5B` |
| Qwen2.5 | `Qwen/Qwen2.5-7B` | `Qwen/Qwen2.5-3B` |
| Qwen3 | `Qwen/Qwen3-8B` | `Qwen/Qwen3-4B` |
| Llama 3 | `meta-llama/Llama-3.1-8B` | `meta-llama/Llama-3.2-3B` |
| OLMo 2 | `allenai/OLMo-2-1124-7B` | `allenai/OLMo-2-0425-1B` |
| SmolLM2 | `HuggingFaceTB/SmolLM2-1.7B` | `HuggingFaceTB/SmolLM2-360M` |

Two-GPU launchers place the target on `cuda:0` and the draft on `cuda:1`.

## Experiments

### Table 2, Fig. 3a: K4V4 across six model pairs

1K-token FineWeb-Edu prompts, three shards of 32 prompts per pair, sequential verifier, 16 generated tokens.

```bash
sbatch --array=0-8 scripts/cross_family/run_sequential_cross_family_spec_1gpu.slurm  # Qwen2.5 3B/1.5B, OLMo 2, SmolLM2
sbatch --array=0-8 scripts/cross_family/run_sequential_cross_family_spec_2gpu.slurm  # Qwen2.5 7B/3B, Qwen3, Llama
SUMMARY_GLOB='outputs/sequential_cross_family_v1/*/ctx_1024/seed_*/summary.json' \
OUT_DIR=outputs/sequential_cross_family_v1/aggregate \
  sbatch scripts/cross_family/aggregate_sequential_spec_runs.slurm
```

The aggregate reports each pair and the equal-model macro (hierarchical bootstrap over pairs, then prompts).

### Section 5.2: draft KL and top-1 agreement

Teacher-forced quality of each draft model with a K4V4 cache. The comparison script also scores the target models, so it needs both grids:

```bash
bash scripts/cross_family/submit_kivi_cross_family.sh               # draft acceptance + quality (also used by Table 3)
bash scripts/cross_family/submit_kivi_cross_family_target_quality.sh
sbatch scripts/cross_family/compare_target_draft_kv_quality.slurm
```

The paper quotes `draft_kl_macro_mean` and `draft_top1_match_macro_mean` from the 1024-token `k4v4` row, frozen in `results/cross-family/target_vs_draft_macro.csv`.

### Table 3, Fig. 2a: per-token vs. grouped per-channel keys

```bash
bash scripts/cross_family/submit_per_token_campaign.sh       # per-token keys, 1-GPU pairs
bash scripts/cross_family/submit_per_token_campaign_2gpu.sh  # per-token keys, 2-GPU pairs
python -m kv_reduce.analysis.aggregate_campaign --results_root outputs/per_token_campaign --out_dir outputs/paper_campaign
sbatch scripts/cross_family/aggregate_kivi_cross_family_meta.slurm   # grouped keys, after submit_kivi_cross_family.sh
python -m kv_reduce.analysis.compare_kv_quantizer_geometry \
  --naive_summary outputs/paper_campaign/campaign_aggregate.json \
  --kivi_preference_csv outputs/kivi_objective_cross_family/meta_aggregate/preference_rows.csv \
  --out_dir outputs/quantizer_geometry
```

The per-token campaign also runs sensitivity, allocation, robustness and long-context stages. Keep them: the aggregator's bootstrap seeds depend on which runs are present, so Table 3's intervals need the full campaign. Frozen results: `results/cross-family/quantizer_geometry_summary.json` and `paired_geometry_results.csv`, with the per-token input `campaign_aggregate.json`.

### Table 4 and Appendix B: key-axis, group-size and residual controls

All on Qwen2.5 3B/1.5B. Table 4 removes the residual window, keeps affine values, and changes only the key quantizer:

```bash
bash scripts/key_geometry/submit_per_token_affine_value_grid.sh  # per-token symmetric keys
bash scripts/key_geometry/submit_kivi_no_residual_grid.sh        # grouped per-channel affine keys
```

Each grid writes its paired K/V contrasts to `spec_aggregate/paired_precision_contrasts.csv` under its `ROOT`.

Group sizes 16/32/64/128 and residual windows 0/32/128/256, each with three seeds:

```bash
sbatch --array=0-11 scripts/key_geometry/run_key_group_geometry_sweep.slurm
sbatch --array=0-3 scripts/key_geometry/aggregate_key_group_geometry.slurm
sbatch scripts/key_geometry/aggregate_key_group_geometry_meta.slurm
# Residual windows: the same three steps with run_key_residual_sweep.slurm,
# aggregate_key_residual_sweep.slurm and aggregate_key_residual_geometry_meta.slurm.
```

### Table 5, Fig. 2b: 16K associative passkey retrieval

Three seeds × 64 examples per model; 16 records per prompt; one-digit distractors.

```bash
bash scripts/passkey/submit_kivi_passkey_powered16k.sh  # grouped keys: Qwen2.5-1.5B, Llama-3.2-3B, Qwen3-4B
bash scripts/passkey/submit_powered_passkey_axis.sh     # per-token keys on Qwen2.5-1.5B, then the axis comparison
```

The axis comparison reads the grouped-key Qwen2.5 run, so start the second chain after the first completes (or pass its last job ID as `DEPENDENCY`).

### Appendix C: LongBench passage retrieval

```bash
bash scripts/longbench/submit_longbench_instruct_cross_family.sh  # Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct, Qwen3-8B
```

`passage_retrieval_en`, three shards of 24 examples per model, middle truncation to 32,752 tokens.

### Table 6: PG-19 at 16K and 32K

Qwen2.5 3B/1.5B; three shards of 12 (16K) or 8 (32K) prompts.

```bash
sbatch --array=0-5 scripts/long_context/run_sequential_pg19_train_long_context_spec.slurm
SUMMARY_GLOB='outputs/sequential_long_context_v2_pg19_train/qwen25_3b_15b/ctx_*/seed_*/summary.json' \
OUT_DIR=outputs/sequential_long_context_v2_pg19_train/qwen25_3b_15b/aggregate \
  sbatch scripts/cross_family/aggregate_sequential_spec_runs.slurm
```

### Appendix D, Fig. 3b: packed storage and Triton kernel

```bash
sbatch scripts/packed/benchmark_packed_kv_attention.slurm         # stored bytes, 1K-32K
sbatch scripts/packed/benchmark_fused_packed_kv_attention.slurm   # Triton kernel vs. rebuild-then-attend vs. BF16 SDPA
```

Frozen summaries: `results/packed/qwen25_15b_summary.json` (storage) and `qwen25_15b_fused_split512_summary.json` (kernel). The paper quotes their 32,768-token `k4v4` rows.

### Appendix E: 256-token continuations and end-to-end prototype

See [REPRODUCING_AXIOM.md](REPRODUCING_AXIOM.md) for commands and [AXIOM_RESULTS.md](AXIOM_RESULTS.md) for results. `python -m kv_reduce.analysis.analyze_axiom_review --check` reproduces every reported number from the released metrics on CPU.

### Appendix F: choosing a precision policy

For each of five model pairs (all except Qwen3), profile per-layer sensitivity under acceptance and under LM quality, build equal-byte allocations from each profile, then cross-evaluate both allocations under both objectives:

```bash
bash scripts/precision_policy/submit_exact_objective_matrix.sh       # mild budgets: 6 and 8 mean bits
bash scripts/precision_policy/submit_aggressive_objective_matrix.sh  # aggressive budgets: 3 and 5 mean bits
python -m kv_reduce.analysis.aggregate_objective_campaign --results_root outputs/objective_kv --out_dir outputs/objective_campaign
```

Defaults are Qwen2.5 3B/1.5B. For other pairs, set `BIG_MODEL`, `SMALL_MODEL`, `NUM_LAYERS` (draft layers) and `OUT_ROOT`; pairs that need two GPUs also need `SPEC_GPUS=2 SMALL_DEVICE=cuda:1`. Name each `OUT_ROOT` after an entry of `MATRIX_LABELS` in the aggregator (e.g. `outputs/objective_kv/olmo2_exact_objective_matrix_v2_bytes`) to get the paper's labels. The aggregate's exactness table also gives the 38,400 target checks cited in Appendix A.

The Spearman correlations come from 20 uniform K/V configurations on Qwen2.5 3B/1.5B at 1K and 4K, reported in `objective_comparison/summary.json`:

```bash
bash scripts/precision_policy/submit_kivi_objective_grid.sh
```

### Introduction: Gaussian KV noise

```bash
python -m kv_reduce.experiments.kv_perturbation_sweep --model Qwen/Qwen2.5-1.5B \
  --dataset_name wikitext --dataset_config wikitext-2-raw-v1 --split validation \
  --seq_len 256 --num_sequences 128 --alphas 0,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1 \
  --perturb_targets keys,values,both --layer_mode all --generate_steps 16 \
  --out_dir outputs/perturb_qwen15b
```

Frozen rows: `results/kv-noise/qwen25_15b_gaussian_all.json`. The paper quotes `top1_match` from the `alpha=0.03` key and value rows. The `alpha=0` rows are a self-check and must show near-zero divergence; otherwise the cache round-trip or decode positions are wrong.

## Outputs

| Entry point | Per-example rows | Summary |
| --- | --- | --- |
| `experiments.benchmark_spec_kv_quantization` (speculative decoding) | `benchmark_rows.csv` | `summary.json` |
| `experiments.profile_kv_quality_sensitivity` (teacher-forced LM quality) | `raw_sequence_rows.csv` | `profile_summary.csv`, `summary.json` |
| `experiments.profile_spec_kv_sensitivity` (per-layer acceptance risk) | `raw_prompt_rows.csv` | `profile_summary.csv`, `summary.json` |
| `experiments.eval_kv_quantized_multiple_choice` (passkey, LongBench) | `example_rows.csv` | `summary.json` |
| `experiments.benchmark_packed_kv_attention` (packed storage) | `results.csv` | `summary.json` |

Each `summary.json` records the run configuration and aggregate metrics. The paper's aggregators check that runs are complete and were produced by the expected evaluator version, e.g. `cached_dynamic_v6_sequential_target` for sequential-verifier speculative decoding and `teacher_forced_cached_v1` for LM quality.

## Implementation notes

- **Quantizer** (`kv_reduce/kv_cache_quantization.py`). `KaVb` means a-bit keys and b-bit values. The paper's layout (`--key_quant_axis per_channel --key_group_size 32 --value_quant_scheme affine`) quantizes keys per channel over groups of 32 tokens and values per token, both affine. The newest `--key_residual_length` keys (default 128) stay in BF16. The flag defaults (`per_token`, `symmetric`) give the naive baseline.
- **Cache semantics.** Only committed tokens enter quantized key groups. Unverified draft tokens stay in the BF16 tail, and rejected suffixes are cropped. The target cache stays in BF16 in every paper experiment.
- **Verification.** `--target_verification_mode sequential` makes one target call per emitted token, and its outputs are checked against target-only greedy decoding. `batched` (the evaluator default) scores a whole proposal in one target call, as the end-to-end prototype does.
- **Bytes** count the quantized payload, BF16 scales and zero-points, and the BF16 residual window. The quality evaluators use fake quantization; `kv_reduce/packed_kv_cache.py` stores real bit-packed tensors.
