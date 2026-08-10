#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs outputs/value_precision_quality_cross_family

EXCLUDE_NODES="${EXCLUDE_NODES:-catalyst-0-9,catalyst-0-15}"
QUALITY_SWEEP_ROOT="${QUALITY_SWEEP_ROOT:-outputs/value_precision_quality_cross_family}"
SPEC_SWEEP_ROOT="${SPEC_SWEEP_ROOT:-outputs/value_precision_cross_family}"
dependency_args=()
if [[ -n "${AFTER_JOBS:-}" ]]; then
  dependency_args+=(--dependency="afterany:${AFTER_JOBS}")
fi

quality_job=$(sbatch --parsable \
  "${dependency_args[@]}" \
  --array="0-17%2" \
  --exclude="$EXCLUDE_NODES" \
  --export="ALL,QUALITY_SWEEP_ROOT=${QUALITY_SWEEP_ROOT},WANDB_PROJECT=${WANDB_PROJECT:-kv-reduce}" \
  scripts/run_value_precision_quality_cross_family.slurm)

aggregate_dependencies="afterok:${quality_job}"
if [[ -n "${SPEC_AGGREGATE_JOB:-}" ]]; then
  aggregate_dependencies+="${aggregate_dependencies:+:}${SPEC_AGGREGATE_JOB}"
fi
aggregate_job=$(sbatch --parsable \
  --dependency="$aggregate_dependencies" \
  --array="0-5%2" \
  --exclude="$EXCLUDE_NODES" \
  --export="ALL,QUALITY_SWEEP_ROOT=${QUALITY_SWEEP_ROOT},SPEC_SWEEP_ROOT=${SPEC_SWEEP_ROOT}" \
  scripts/aggregate_value_precision_quality_cross_family.slurm)

printf 'Submitted cross-family matched quality sweep\n  quality: %s\n  aggregate: %s\n' \
  "$quality_job" "$aggregate_job"
