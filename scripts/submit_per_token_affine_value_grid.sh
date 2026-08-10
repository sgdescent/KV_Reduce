#!/bin/bash
set -euo pipefail

dependency_args=()
if [[ -n "${AFTER_JOB:-}" ]]; then
  dependency_args+=(--dependency="afterok:${AFTER_JOB}")
fi

root="${ROOT:-outputs/quantizer_factorial/per_token_key_affine_value/qwen25_3b_15b}"
exclude="${EXCLUDE_NODES:-catalyst-0-9,catalyst-0-15}"
configs="${QUANT_CONFIGS:-none;k16v8;k16v4;k16v3;k16v2;k8v8;k8v4;k8v3;k8v2;k4v8;k4v4;k4v3;k4v2;k3v8;k3v4;k3v3;k3v2;k2v8;k2v4;k2v3;k2v2}"

spec_job=$(sbatch --parsable \
  "${dependency_args[@]}" \
  --array=0-5%1 \
  --exclude="$exclude" \
  --export="ALL,KEY_QUANT_AXIS=per_token,KEY_RESIDUAL_LENGTH=0,VALUE_QUANT_SCHEME=affine,SPEC_ROOT=${root}/spec,QUANT_CONFIGS=${configs},WANDB_PROJECT=kv-reduce,WANDB_GROUP=quantizer-factorial-per-token-affine-spec" \
  scripts/run_kivi_objective_grid_spec.slurm)

quality_job=$(sbatch --parsable \
  "${dependency_args[@]}" \
  --array=0-5%1 \
  --exclude="$exclude" \
  --export="ALL,KEY_QUANT_AXIS=per_token,KEY_RESIDUAL_LENGTH=0,VALUE_QUANT_SCHEME=affine,QUALITY_ROOT=${root}/quality,QUANT_CONFIGS=${configs},WANDB_PROJECT=kv-reduce,WANDB_GROUP=quantizer-factorial-per-token-affine-quality" \
  scripts/run_kivi_objective_grid_quality.slurm)

aggregate_job=$(sbatch --parsable \
  --dependency="afterok:${spec_job}:${quality_job}" \
  --exclude="$exclude" \
  --export="ALL,ROOT=${root}" \
  scripts/aggregate_kivi_objective_grid.slurm)

printf 'Submitted per-token-key/affine-value factorial grid\n  speculative: %s\n  quality: %s\n  aggregate: %s\n' \
  "$spec_job" "$quality_job" "$aggregate_job"
