#!/bin/bash
set -euo pipefail

dependency_args=()
if [[ -n "${AFTER_JOB:-}" ]]; then
  dependency_args+=(--dependency="afterok:${AFTER_JOB}")
fi

root="${ROOT:-outputs/kivi_objective_grid_no_residual/qwen25_3b_15b}"
exclude="${EXCLUDE_NODES:-catalyst-0-9,catalyst-0-15}"
configs="${QUANT_CONFIGS:-none;k8v4;k4v8;k8v3;k3v8;k4v3;k3v4;k4v2;k2v4;k3v2;k2v3}"

spec_job=$(sbatch --parsable \
  "${dependency_args[@]}" \
  --array=0-5%1 \
  --exclude="$exclude" \
  --export="ALL,KEY_RESIDUAL_LENGTH=0,SPEC_ROOT=${root}/spec,QUANT_CONFIGS=${configs},WANDB_PROJECT=kv-reduce,WANDB_GROUP=kivi-objective-no-residual-spec" \
  scripts/run_kivi_objective_grid_spec.slurm)

quality_job=$(sbatch --parsable \
  "${dependency_args[@]}" \
  --array=0-5%1 \
  --exclude="$exclude" \
  --export="ALL,KEY_RESIDUAL_LENGTH=0,QUALITY_ROOT=${root}/quality,QUANT_CONFIGS=${configs},WANDB_PROJECT=kv-reduce,WANDB_GROUP=kivi-objective-no-residual-quality" \
  scripts/run_kivi_objective_grid_quality.slurm)

aggregate_job=$(sbatch --parsable \
  --dependency="afterok:${spec_job}:${quality_job}" \
  --exclude="$exclude" \
  --export="ALL,ROOT=${root}" \
  scripts/aggregate_kivi_objective_grid.slurm)

printf 'Submitted zero-residual matched KIVI objective grid\n  speculative: %s\n  quality: %s\n  aggregate: %s\n' \
  "$spec_job" "$quality_job" "$aggregate_job"
