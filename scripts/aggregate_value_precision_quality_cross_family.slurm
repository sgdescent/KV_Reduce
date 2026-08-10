#!/bin/bash
#SBATCH --job-name=kv_value_q_xfam_agg
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
quality_root="${QUALITY_SWEEP_ROOT:-outputs/value_precision_quality_cross_family}/$pair"
spec_root="${SPEC_SWEEP_ROOT:-outputs/value_precision_cross_family}/$pair"

python3 -u aggregate_value_precision_quality_sweep.py \
  --sweep_dir "$quality_root" \
  --out_dir "$quality_root/aggregate"

python3 -u compare_value_precision_objectives.py \
  --spec_summary "$spec_root/aggregate/summary.json" \
  --quality_summary "$quality_root/aggregate/summary.json" \
  --out_dir "$quality_root/objective_comparison"
