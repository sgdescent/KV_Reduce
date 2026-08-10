#!/bin/bash
set -euo pipefail

dependency_args=()
if [[ -n "${AFTER_JOB:-}" ]]; then
  dependency_args+=(--dependency="afterok:${AFTER_JOB}")
fi
exclude="${EXCLUDE_NODES:-catalyst-0-9,catalyst-0-15}"

spec=$(sbatch --parsable \
  "${dependency_args[@]}" \
  --array=0-5%1 \
  --exclude="$exclude" \
  --export=ALL,WANDB_PROJECT=kv-reduce \
  scripts/run_kivi_long_context_spec.slurm)
quality=$(sbatch --parsable \
  "${dependency_args[@]}" \
  --array=0-5%1 \
  --exclude="$exclude" \
  --export=ALL,WANDB_PROJECT=kv-reduce \
  scripts/run_kivi_long_context_quality.slurm)
aggregate=$(sbatch --parsable \
  --dependency="afterok:${spec}:${quality}" \
  --exclude="$exclude" \
  scripts/aggregate_kivi_long_context.slurm)

printf 'Submitted long-context KIVI objective campaign\n  speculative: %s\n  quality: %s\n  aggregate: %s\n' \
  "$spec" "$quality" "$aggregate"
