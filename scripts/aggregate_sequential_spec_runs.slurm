#!/bin/bash
#SBATCH --job-name=kv_seq_specagg
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --exclude=catalyst-0-[9,15]
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "${REPO_ROOT:-/home/sakshamg/KV_Reduce}"
if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

full_run_args=(--require_full_runs)
if [[ "${ALLOW_UNDERFILLED_RUNS:-0}" == "1" ]]; then
  full_run_args=()
fi

python3 -u aggregate_sequential_spec_runs.py \
  --summary_glob "${SUMMARY_GLOB:?Set SUMMARY_GLOB}" \
  --out_dir "${OUT_DIR:?Set OUT_DIR}" \
  --bootstrap_samples "${BOOTSTRAP_SAMPLES:-20000}" \
  "${full_run_args[@]}"
