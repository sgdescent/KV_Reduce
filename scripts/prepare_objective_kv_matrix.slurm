#!/bin/bash
#SBATCH --job-name=kv_obj_prepare
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

python3 -u prepare_objective_kv_matrix.py \
  --quality_profile_csv "${QUALITY_PROFILE_CSV:?Set QUALITY_PROFILE_CSV}" \
  --acceptance_profile_csv "${ACCEPTANCE_PROFILE_CSV:?Set ACCEPTANCE_PROFILE_CSV}" \
  --num_layers "${NUM_LAYERS:?Set NUM_LAYERS}" \
  --budgets "${BUDGETS:-6,8,10,12}" \
  --contexts "${CONTEXTS:-512,1024,4096}" \
  --seeds "${SEEDS:-0,1,2}" \
  --num_eval "${NUM_EVAL:-32}" \
  --out_dir "${OUT_DIR:?Set OUT_DIR}"
