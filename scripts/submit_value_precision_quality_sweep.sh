#!/bin/bash
set -euo pipefail

mkdir -p logs outputs
dependency_args=()
if [[ -n "${AFTER_JOB:-}" ]]; then
  dependency_args+=(--dependency="afterok:${AFTER_JOB}")
fi

array_job=$(sbatch --parsable \
  "${dependency_args[@]}" \
  --array=0-8%1 \
  --exclude=catalyst-0-9,catalyst-0-15 \
  --export=ALL,SWEEP_ROOT="${SWEEP_ROOT:-outputs/value_precision_quality/qwen25_15b}",WANDB_PROJECT="${WANDB_PROJECT:-kv-reduce}" \
  scripts/run_value_precision_quality_sweep.slurm)

aggregate_job=$(sbatch --parsable \
  --dependency="afterok:${array_job}" \
  --exclude=catalyst-0-9,catalyst-0-15 \
  --export=ALL,SWEEP_ROOT="${SWEEP_ROOT:-outputs/value_precision_quality/qwen25_15b}",OUT_DIR="${OUT_DIR:-outputs/value_precision_quality/qwen25_15b/aggregate}" \
  scripts/aggregate_value_precision_quality_sweep.slurm)

printf 'Submitted matched teacher-forced value sweep\n  array: %s\n  aggregate: %s\n' "$array_job" "$aggregate_job"
