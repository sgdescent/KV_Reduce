#!/bin/bash
#SBATCH --job-name=kv_value_xfam_agg
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

pairs=(
  qwen25_3b_15b
  qwen25_7b_3b
  qwen3_8b_4b
  llama31_8b_llama32_3b
  olmo2_7b_1b
  smollm2_17b_360m
)
pair="${pairs[${SLURM_ARRAY_TASK_ID:-0}]}"
root="${VALUE_SWEEP_ROOT:-outputs/value_precision_cross_family}/$pair"

python3 -u aggregate_value_precision_sweep.py \
  --sweep_dir "$root" \
  --out_dir "$root/aggregate"
