#!/bin/bash
set -euo pipefail

cd "${REPO_ROOT:-/home/sakshamg/KV_Reduce}"
mkdir -p logs

submit_args=(--parsable --array=0-3%1 --exclude=catalyst-0-9,catalyst-0-15)
if [[ "${START_HELD:-1}" == "1" ]]; then
  submit_args+=(--hold)
fi

matrix_job=$(sbatch "${submit_args[@]}" \
  --export=ALL \
  scripts/run_model_verifier_exactness_matrix.slurm)
aggregate_job=$(sbatch --parsable \
  --dependency="afterok:$matrix_job" \
  --export=ALL \
  scripts/aggregate_verifier_exactness_matrix.slurm)

printf 'Submitted verifier exactness matrix\n  matrix: %s\n  aggregate: %s\n  output: %s\n' \
  "$matrix_job" "$aggregate_job" "${OUT_ROOT:-outputs/target_verification_diagnostic/model_matrix}"
