#!/bin/bash
#SBATCH --job-name=kv_obj_matrix_agg
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
if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi
python3 -u aggregate_objective_kv_matrix.py \
  --matrix_dir "${MATRIX_DIR:?Set MATRIX_DIR}" \
  --out_dir "${OUT_DIR:?Set OUT_DIR}"
