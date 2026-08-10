#!/bin/bash
set -euo pipefail

dependency_args=()
if [[ -n "${AFTER_JOB:-}" ]]; then
  dependency_args+=(--dependency="afterok:${AFTER_JOB}")
fi

spec_job=$(sbatch --parsable \
  "${dependency_args[@]}" \
  --array=0-5%1 \
  --exclude=catalyst-0-9,catalyst-0-15 \
  --export=ALL,WANDB_PROJECT=kv-reduce \
  scripts/run_kivi_objective_grid_spec.slurm)

quality_job=$(sbatch --parsable \
  "${dependency_args[@]}" \
  --array=0-5%1 \
  --exclude=catalyst-0-9,catalyst-0-15 \
  --export=ALL,WANDB_PROJECT=kv-reduce \
  scripts/run_kivi_objective_grid_quality.slurm)

aggregate_job=$(sbatch --parsable \
  --dependency="afterok:${spec_job}:${quality_job}" \
  --exclude=catalyst-0-9,catalyst-0-15 \
  scripts/aggregate_kivi_objective_grid.slurm)

printf 'Submitted matched KIVI objective grid\n  speculative: %s\n  quality: %s\n  aggregate: %s\n' \
  "$spec_job" "$quality_job" "$aggregate_job"
