#!/bin/bash
#SBATCH --job-name=kv_obj_final
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs outputs
if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

python3 -u aggregate_objective_kv_results.py \
  --quality_summary "${QUALITY_SUMMARY:?Set QUALITY_SUMMARY}" \
  --acceptance_summary "${ACCEPTANCE_SUMMARY:?Set ACCEPTANCE_SUMMARY}" \
  --quality_allocation "${QUALITY_ALLOCATION:?Set QUALITY_ALLOCATION}" \
  --acceptance_allocation "${ACCEPTANCE_ALLOCATION:?Set ACCEPTANCE_ALLOCATION}" \
  --comparison_summary "${COMPARISON_SUMMARY:?Set COMPARISON_SUMMARY}" \
  --out_dir "${OUT_DIR:-outputs/kv_objective_final_${SLURM_JOB_ID:-local}}"
