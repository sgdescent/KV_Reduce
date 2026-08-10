#!/bin/bash
set -euo pipefail

dependency_args=()
if [[ -n "${AFTER_JOB:-}" ]]; then
  dependency_args+=(--dependency="afterok:${AFTER_JOB}")
fi
exclude="${EXCLUDE_NODES:-catalyst-0-9,catalyst-0-15}"
root="${SWEEP_ROOT:-outputs/kivi_reversal_powered/qwen25_7b_3b}"

spec=$(sbatch --parsable \
  "${dependency_args[@]}" \
  --array="0-2%1" \
  --exclude="$exclude" \
  --export="ALL,SWEEP_ROOT=${root},WANDB_PROJECT=${WANDB_PROJECT:-kv-reduce}" \
  scripts/run_qwen7_reversal_powered_spec.slurm)
quality=$(sbatch --parsable \
  "${dependency_args[@]}" \
  --array="0-2%1" \
  --exclude="$exclude" \
  --export="ALL,SWEEP_ROOT=${root},WANDB_PROJECT=${WANDB_PROJECT:-kv-reduce}" \
  scripts/run_qwen7_reversal_powered_quality.slurm)
aggregate=$(sbatch --parsable \
  --dependency="afterok:${spec}:${quality}" \
  --exclude="$exclude" \
  --export="ALL,SWEEP_ROOT=${root}" \
  scripts/aggregate_qwen7_reversal_powered.slurm)

printf 'Submitted powered Qwen2.5-7B/3B reversal replication\n  spec: %s\n  quality: %s\n  aggregate: %s\n' \
  "$spec" "$quality" "$aggregate"
