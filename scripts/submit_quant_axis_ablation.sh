#!/bin/bash
set -euo pipefail

dependency_args=()
if [[ -n "${AFTER_JOB:-}" ]]; then
  dependency_args+=(--dependency="afterok:${AFTER_JOB}")
fi

r0_job=$(sbatch --parsable \
  "${dependency_args[@]}" \
  --array=0-5%1 \
  --exclude=catalyst-0-9,catalyst-0-15 \
  --export=ALL,AXIS_VARIANT=per_channel_affine_r0,KEY_QUANT_AXIS=per_channel,KEY_GROUP_SIZE=32,KEY_RESIDUAL_LENGTH=0,WANDB_PROJECT=kv-reduce \
  scripts/run_quant_axis_ablation.slurm)

r128_job=$(sbatch --parsable \
  "${dependency_args[@]}" \
  --array=0-5%1 \
  --exclude=catalyst-0-9,catalyst-0-15 \
  --export=ALL,AXIS_VARIANT=per_channel_affine_r128,KEY_QUANT_AXIS=per_channel,KEY_GROUP_SIZE=32,KEY_RESIDUAL_LENGTH=128,WANDB_PROJECT=kv-reduce \
  scripts/run_quant_axis_ablation.slurm)

aggregate_job=$(sbatch --parsable \
  --dependency="afterok:${r0_job}:${r128_job}" \
  --exclude=catalyst-0-9,catalyst-0-15 \
  scripts/aggregate_quant_axis_ablation.slurm)

printf 'Submitted key-axis ablation\n  per-channel r0: %s\n  per-channel r128: %s\n  comparison: %s\n' \
  "$r0_job" "$r128_job" "$aggregate_job"
