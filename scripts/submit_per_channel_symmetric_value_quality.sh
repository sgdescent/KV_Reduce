#!/bin/bash
set -euo pipefail

dependency_args=()
if [[ -n "${AFTER_JOB:-}" ]]; then
  dependency_args+=(--dependency="afterok:${AFTER_JOB}")
fi

residual_length="${RESIDUAL_LENGTH:-128}"
root="${ROOT:-outputs/quantizer_factorial/per_channel_key_symmetric_value_r${residual_length}/qwen25_3b_15b}"
exclude="${EXCLUDE_NODES:-catalyst-0-9,catalyst-0-15}"
configs="${QUANT_CONFIGS:-none;k16v8;k16v4;k16v3;k16v2;k8v8;k8v4;k8v3;k8v2;k4v8;k4v4;k4v3;k4v2;k3v8;k3v4;k3v3;k3v2;k2v8;k2v4;k2v3;k2v2}"

quality_job=$(sbatch --parsable \
  "${dependency_args[@]}" \
  --array=0-5%1 \
  --exclude="$exclude" \
  --export="ALL,KEY_QUANT_AXIS=per_channel,KEY_GROUP_SIZE=32,KEY_RESIDUAL_LENGTH=${residual_length},VALUE_QUANT_SCHEME=symmetric,QUALITY_ROOT=${root}/quality,QUANT_CONFIGS=${configs},WANDB_PROJECT=kv-reduce,WANDB_GROUP=quantizer-factorial-per-channel-symmetric-r${residual_length}-quality" \
  scripts/run_kivi_objective_grid_quality.slurm)

aggregate_job=$(sbatch --parsable \
  --dependency="afterok:${quality_job}" \
  --exclude="$exclude" \
  --export="ALL,SWEEP_DIR=${root}/quality,OUT_DIR=${root}/quality_aggregate" \
  scripts/aggregate_kivi_quality_grid.slurm)

printf 'Submitted per-channel-key/symmetric-value quality grid (residual=%s)\n  quality: %s\n  aggregate: %s\n' \
  "$residual_length" "$quality_job" "$aggregate_job"
