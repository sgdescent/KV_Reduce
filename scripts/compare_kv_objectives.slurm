#!/bin/bash
#SBATCH --job-name=kv_obj_compare
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

QUALITY_PROFILE_CSV="${QUALITY_PROFILE_CSV:?Set QUALITY_PROFILE_CSV}"
ACCEPTANCE_PROFILE_CSV="${ACCEPTANCE_PROFILE_CSV:?Set ACCEPTANCE_PROFILE_CSV}"
QUALITY_ALLOCATION="${QUALITY_ALLOCATION:-}"
ACCEPTANCE_ALLOCATION="${ACCEPTANCE_ALLOCATION:-}"
OUT_DIR="${OUT_DIR:-outputs/kv_objective_comparison_${SLURM_JOB_ID:-local}}"

cmd=(
  python3 -u compare_kv_objectives.py
  --quality_profile_csv "$QUALITY_PROFILE_CSV"
  --acceptance_profile_csv "$ACCEPTANCE_PROFILE_CSV"
  --out_dir "$OUT_DIR"
)
if [[ -n "$QUALITY_ALLOCATION" && -n "$ACCEPTANCE_ALLOCATION" ]]; then
  cmd+=(--quality_allocation "$QUALITY_ALLOCATION" --acceptance_allocation "$ACCEPTANCE_ALLOCATION")
fi
"${cmd[@]}"
