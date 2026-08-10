#!/bin/bash
set -euo pipefail

dependency="${DEPENDENCY:-}"
dependency_args=()
if [[ -n "$dependency" ]]; then
  dependency_args+=(--dependency="afterok:${dependency}")
fi

array_job=$(sbatch --parsable \
  "${dependency_args[@]}" \
  --array=0-8%1 \
  --exclude=catalyst-0-9,catalyst-0-15 \
  --export="ALL,WANDB_PROJECT=${WANDB_PROJECT:-kv-reduce}" \
  scripts/run_kivi_passkey.slurm)
aggregate_job=$(sbatch --parsable \
  --dependency="afterok:${array_job}" \
  --exclude=catalyst-0-9,catalyst-0-15 \
  scripts/aggregate_kivi_passkey.slurm)

printf 'passkey_job=%s\n' "$array_job"
printf 'passkey_aggregate_job=%s\n' "$aggregate_job"
