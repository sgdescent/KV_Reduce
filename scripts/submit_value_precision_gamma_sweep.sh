#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs outputs/value_precision_gamma
dependency_args=()
if [[ -n "${AFTER_JOB:-}" ]]; then
  dependency_args+=(--dependency="afterok:${AFTER_JOB}")
fi

array_job=$(sbatch --parsable \
  "${dependency_args[@]}" \
  --array="0-8%2" \
  --exclude="${EXCLUDE_NODES:-catalyst-0-9,catalyst-0-15}" \
  --export="ALL,SWEEP_ROOT=${SWEEP_ROOT:-outputs/value_precision_gamma/qwen25_3b_15b},WANDB_PROJECT=${WANDB_PROJECT:-kv-reduce}" \
  scripts/run_value_precision_gamma_sweep.slurm)

aggregate_job=$(sbatch --parsable \
  --dependency="afterok:${array_job}" \
  --exclude="${EXCLUDE_NODES:-catalyst-0-9,catalyst-0-15}" \
  --export="ALL,SWEEP_ROOT=${SWEEP_ROOT:-outputs/value_precision_gamma/qwen25_3b_15b}" \
  scripts/aggregate_value_precision_gamma_sweep.slurm)

printf 'Submitted value-precision gamma sweep\n  array: %s\n  aggregate: %s\n' \
  "$array_job" "$aggregate_job"
