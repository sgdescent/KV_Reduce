#!/bin/bash
set -euo pipefail

dependency_args=()
if [[ -n "${AFTER_JOB:-}" ]]; then
  dependency_args+=(--dependency="afterok:${AFTER_JOB}")
fi
exclude="${EXCLUDE_NODES:-}"
concurrency="${CAMPAIGN_CONCURRENCY:-1}"

quality=$(sbatch --parsable \
  "${dependency_args[@]}" \
  --array="0-11%${concurrency}" \
  ${exclude:+"--exclude=$exclude"} \
  --export=ALL,WANDB_PROJECT=kv-reduce \
  scripts/cross_family/run_kivi_cross_family_target_quality.slurm)
aggregate=$(sbatch --parsable \
  --dependency="afterok:${quality}" \
  --array="0-5%2" \
  ${exclude:+"--exclude=$exclude"} \
  scripts/cross_family/aggregate_kivi_cross_family_target_quality.slurm)

printf 'Submitted cross-family target-model KIVI quality campaign\n  quality: %s\n  aggregate: %s\n' \
  "$quality" "$aggregate"
