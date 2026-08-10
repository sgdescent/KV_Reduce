#!/bin/bash
set -euo pipefail

dependency_args=()
if [[ -n "${AFTER_JOB:-}" ]]; then
  dependency_args+=(--dependency="afterok:${AFTER_JOB}")
fi
exclude="${EXCLUDE_NODES:-catalyst-0-9,catalyst-0-15}"
root="${SWEEP_ROOT:-outputs/kivi_gamma/qwen25_3b_15b}"
concurrency="${CAMPAIGN_CONCURRENCY:-1}"

array_job=$(sbatch --parsable \
  "${dependency_args[@]}" \
  --array="0-8%${concurrency}" \
  --exclude="$exclude" \
  --export="ALL,SWEEP_ROOT=${root},WANDB_PROJECT=${WANDB_PROJECT:-kv-reduce}" \
  scripts/run_kivi_gamma_sweep.slurm)
aggregate_job=$(sbatch --parsable \
  --dependency="afterok:${array_job}" \
  --exclude="$exclude" \
  --export="ALL,SWEEP_ROOT=${root}" \
  scripts/aggregate_value_precision_gamma_sweep.slurm)

printf 'Submitted matched-KIVI gamma sweep\n  array: %s\n  aggregate: %s\n' \
  "$array_job" "$aggregate_job"
