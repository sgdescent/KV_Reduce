#!/bin/bash
#SBATCH --job-name=kv_value_gamma_agg
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

root="${SWEEP_ROOT:-outputs/value_precision_gamma/qwen25_3b_15b}"
python3 -u aggregate_value_precision_gamma_sweep.py \
  --sweep_dir "$root" \
  --out_dir "$root/aggregate"
