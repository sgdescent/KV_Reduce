#!/bin/bash
set -euo pipefail

exclude="${EXCLUDE_NODES:-catalyst-0-9,catalyst-0-15}"
root="${ROOT:-outputs/kivi_joint_target_draft/qwen25_3b_15b}"
dependency_args=()
if [[ -n "${AFTER_JOB:-}" ]]; then
  dependency_args+=(--dependency="afterok:${AFTER_JOB}")
fi

grid_job=$(sbatch --parsable \
  "${dependency_args[@]}" \
  --array="0-5%${ARRAY_THROTTLE:-1}" \
  --exclude="$exclude" \
  --export="ALL,ROOT=${root},WANDB_PROJECT=kv-reduce" \
  scripts/run_joint_target_draft_grid.slurm)
aggregate_job=$(sbatch --parsable \
  --dependency="afterok:${grid_job}" \
  --exclude="$exclude" \
  --export="ALL,ROOT=${root}" \
  scripts/aggregate_joint_target_draft_grid.slurm)

printf 'Submitted joint target/draft KIVI grid\n  grid: %s\n  aggregate: %s\n' \
  "$grid_job" "$aggregate_job"
