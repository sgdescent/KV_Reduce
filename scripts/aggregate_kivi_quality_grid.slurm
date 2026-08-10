#!/bin/bash
#SBATCH --job-name=kv_kivi_quality_agg
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "${REPO_ROOT:-/home/sakshamg/KV_Reduce}"
if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

python3 -u aggregate_value_precision_quality_sweep.py \
  --sweep_dir "${SWEEP_DIR:?SWEEP_DIR is required}" \
  --out_dir "${OUT_DIR:?OUT_DIR is required}"
