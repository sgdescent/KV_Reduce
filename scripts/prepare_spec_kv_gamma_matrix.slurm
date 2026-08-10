#!/bin/bash
#SBATCH --job-name=kv_gamma_prep
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

python3 -u prepare_spec_kv_gamma_matrix.py \
  --allocation_root "${ALLOCATION_ROOT:?Set ALLOCATION_ROOT}" \
  --budgets "${BUDGETS:-6,8}" \
  --contexts "${CONTEXTS:-1024,4096}" \
  --seeds "${SEEDS:-0,1,2}" \
  --draft_steps "${DRAFT_STEPS_LIST:-2,4,8}" \
  --num_eval "${NUM_EVAL:-32}" \
  --out_dir "${OUT_DIR:?Set OUT_DIR}"
