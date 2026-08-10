#!/bin/bash
set -euo pipefail

dependency_args=()
if [[ -n "${AFTER_JOB:-}" ]]; then
  dependency_args+=(--dependency="afterok:${AFTER_JOB}")
fi
exclude="${EXCLUDE_NODES:-catalyst-0-9,catalyst-0-15}"
root="${OUT_ROOT:-outputs/target_verification_diagnostic/powered_backend_dtype}"

runs=$(sbatch --parsable \
  "${dependency_args[@]}" \
  --array="0-3%1" \
  --exclude="$exclude" \
  --export="ALL,OUT_ROOT=${root}" \
  scripts/run_verifier_exactness_replication.slurm)
aggregate=$(sbatch --parsable \
  --dependency="afterok:${runs}" \
  --exclude="$exclude" \
  --export="ALL,OUT_ROOT=${root}" \
  scripts/aggregate_verifier_exactness_replication.slurm)

printf 'Submitted powered verifier exactness replication\n  runs: %s\n  aggregate: %s\n' \
  "$runs" "$aggregate"
