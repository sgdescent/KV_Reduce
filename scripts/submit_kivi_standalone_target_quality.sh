#!/bin/bash
set -euo pipefail

dependency_args=()
if [[ -n "${AFTER_JOB:-}" ]]; then
  dependency_args+=(--dependency="afterok:${AFTER_JOB}")
fi

root="${ROOT:-outputs/kivi_standalone_quality/qwen25_3b}"
model="${MODEL:-Qwen/Qwen2.5-3B}"
exclude="${EXCLUDE_NODES:-catalyst-0-9,catalyst-0-15}"

quality_job=$(sbatch --parsable \
  "${dependency_args[@]}" \
  --array=0-5%1 \
  --exclude="$exclude" \
  --export="ALL,SMALL_MODEL=${model},QUALITY_ROOT=${root}/quality,WANDB_PROJECT=kv-reduce,WANDB_GROUP=kivi-standalone-target-quality" \
  scripts/run_kivi_objective_grid_quality.slurm)

aggregate_job=$(sbatch --parsable \
  --dependency="afterok:${quality_job}" \
  --exclude="$exclude" \
  --export="ALL,SWEEP_DIR=${root}/quality,OUT_DIR=${root}/aggregate" \
  scripts/aggregate_kivi_quality_grid.slurm)

printf 'Submitted standalone target-model KIVI quality grid\n  quality: %s\n  aggregate: %s\n' \
  "$quality_job" "$aggregate_job"
