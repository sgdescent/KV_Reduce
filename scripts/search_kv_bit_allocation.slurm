#!/bin/bash
#SBATCH --job-name=kv_bit_search
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

VENV_PATH="${VENV_PATH:-}"
if [[ -z "$VENV_PATH" ]]; then
  for candidate in .venv venv env; do
    if [[ -f "$candidate/bin/activate" ]]; then
      VENV_PATH="$candidate"
      break
    fi
  done
fi
if [[ -n "$VENV_PATH" ]]; then
  # shellcheck disable=SC1090
  source "$VENV_PATH/bin/activate"
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
PROFILE_CSV="${PROFILE_CSV:?Set PROFILE_CSV to the profile_summary.csv from profile_spec_kv_sensitivity.py}"
NUM_LAYERS="${NUM_LAYERS:-28}"
ALLOWED_BITS="${ALLOWED_BITS:-4,8,16}"
MAX_COMPONENT_DROP="${MAX_COMPONENT_DROP:-0.01}"
MAX_TOTAL_DROP="${MAX_TOTAL_DROP:-0.05}"
NAME="${NAME:-sensitivity_aware}"
OUT_DIR="${OUT_DIR:-outputs/spec_kv_allocation_${SLURM_JOB_ID:-local}}"

"$PYTHON_BIN" -u search_kv_bit_allocation.py \
  --profile_csv "$PROFILE_CSV" \
  --num_layers "$NUM_LAYERS" \
  --allowed_bits "$ALLOWED_BITS" \
  --max_component_drop "$MAX_COMPONENT_DROP" \
  --max_total_drop "$MAX_TOTAL_DROP" \
  --name "$NAME" \
  --out_dir "$OUT_DIR"
