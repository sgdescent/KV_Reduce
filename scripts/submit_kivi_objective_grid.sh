#!/bin/bash
set -euo pipefail

spec_gpus="${SPEC_GPUS:-1}"
spec_throttle="${SPEC_ARRAY_THROTTLE:-1}"
quality_throttle="${QUALITY_ARRAY_THROTTLE:-1}"
small_device="${SMALL_DEVICE:-cuda:0}"

if [[ ! "$spec_gpus" =~ ^[1-9][0-9]*$ ]]; then
  echo "SPEC_GPUS must be a positive integer, got: $spec_gpus" >&2
  exit 2
fi
if [[ ! "$spec_throttle" =~ ^[1-9][0-9]*$ ]]; then
  echo "SPEC_ARRAY_THROTTLE must be a positive integer, got: $spec_throttle" >&2
  exit 2
fi
if [[ ! "$quality_throttle" =~ ^[1-9][0-9]*$ ]]; then
  echo "QUALITY_ARRAY_THROTTLE must be a positive integer, got: $quality_throttle" >&2
  exit 2
fi
if [[ "$small_device" == "cuda:1" && "$spec_gpus" -lt 2 ]]; then
  echo "SMALL_DEVICE=cuda:1 requires SPEC_GPUS>=2." >&2
  exit 2
fi

dependency_args=()
if [[ -n "${AFTER_JOB:-}" ]]; then
  dependency_args+=(--dependency="afterok:${AFTER_JOB}")
fi

spec_job=$(sbatch --parsable \
  "${dependency_args[@]}" \
  --array="0-5%${spec_throttle}" \
  --gres="gpu:${spec_gpus}" \
  --exclude=catalyst-0-9,catalyst-0-15 \
  --export=ALL,WANDB_PROJECT=kv-reduce \
  scripts/run_kivi_objective_grid_spec.slurm)

quality_job=$(sbatch --parsable \
  "${dependency_args[@]}" \
  --array="0-5%${quality_throttle}" \
  --exclude=catalyst-0-9,catalyst-0-15 \
  --export=ALL,WANDB_PROJECT=kv-reduce \
  scripts/run_kivi_objective_grid_quality.slurm)

aggregate_job=$(sbatch --parsable \
  --dependency="afterok:${spec_job}:${quality_job}" \
  --exclude=catalyst-0-9,catalyst-0-15 \
  scripts/aggregate_kivi_objective_grid.slurm)

printf 'Submitted matched KIVI objective grid\n  speculative: %s\n  quality: %s\n  aggregate: %s\n' \
  "$spec_job" "$quality_job" "$aggregate_job"
