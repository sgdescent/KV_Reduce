#!/bin/bash
set -euo pipefail

root="${ROOT:-outputs/kivi_group_residual_sweep/qwen25_3b_15b}"
exclude="${EXCLUDE_NODES:-catalyst-0-9,catalyst-0-15}"
configs="${QUANT_CONFIGS:-none;k8v8;k8v4;k4v8;k4v4;k3v8}"
groups=(16 32 64 128)
residuals=(0 128)
chain_job="${AFTER_JOB:-}"

for group_size in "${groups[@]}"; do
  for residual_length in "${residuals[@]}"; do
    dependency_args=()
    if [[ -n "$chain_job" ]]; then
      dependency_args+=(--dependency="afterok:${chain_job}")
    fi
    cell_root="${root}/group_${group_size}/residual_${residual_length}"
    group_name="kivi-g${group_size}-r${residual_length}"

    spec_job=$(sbatch --parsable \
      "${dependency_args[@]}" \
      --array=0-2%1 \
      --exclude="$exclude" \
      --export="ALL,KEY_QUANT_AXIS=per_channel,KEY_GROUP_SIZE=${group_size},KEY_RESIDUAL_LENGTH=${residual_length},VALUE_QUANT_SCHEME=affine,SPEC_ROOT=${cell_root}/spec,QUANT_CONFIGS=${configs},WANDB_PROJECT=kv-reduce,WANDB_GROUP=${group_name}-spec" \
      scripts/run_kivi_objective_grid_spec.slurm)

    quality_job=$(sbatch --parsable \
      "${dependency_args[@]}" \
      --array=0-2%1 \
      --exclude="$exclude" \
      --export="ALL,KEY_QUANT_AXIS=per_channel,KEY_GROUP_SIZE=${group_size},KEY_RESIDUAL_LENGTH=${residual_length},VALUE_QUANT_SCHEME=affine,QUALITY_ROOT=${cell_root}/quality,QUANT_CONFIGS=${configs},WANDB_PROJECT=kv-reduce,WANDB_GROUP=${group_name}-quality" \
      scripts/run_kivi_objective_grid_quality.slurm)

    chain_job=$(sbatch --parsable \
      --dependency="afterok:${spec_job}:${quality_job}" \
      --exclude="$exclude" \
      --export="ALL,ROOT=${cell_root}" \
      scripts/aggregate_kivi_objective_grid.slurm)

    printf 'Queued group=%s residual=%s: spec=%s quality=%s aggregate=%s\n' \
      "$group_size" "$residual_length" "$spec_job" "$quality_job" "$chain_job"
  done
done

printf 'Final group/residual sweep dependency: %s\n' "$chain_job"
