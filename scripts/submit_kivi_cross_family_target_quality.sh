#!/bin/bash
set -euo pipefail

dependency_args=()
if [[ -n "${AFTER_JOB:-}" ]]; then
  dependency_args+=(--dependency="afterok:${AFTER_JOB}")
fi
exclude="${EXCLUDE_NODES:-catalyst-0-9,catalyst-0-15}"
concurrency="${CAMPAIGN_CONCURRENCY:-1}"

quality=$(sbatch --parsable \
  "${dependency_args[@]}" \
  --array="0-11%${concurrency}" \
  --exclude="$exclude" \
  --export=ALL,WANDB_PROJECT=kv-reduce \
  scripts/run_kivi_cross_family_target_quality.slurm)
aggregate=$(sbatch --parsable \
  --dependency="afterok:${quality}" \
  --array="0-5%2" \
  --exclude="$exclude" \
  scripts/aggregate_kivi_cross_family_target_quality.slurm)

printf 'Submitted cross-family target-model KIVI quality campaign\n  quality: %s\n  aggregate: %s\n' \
  "$quality" "$aggregate"
