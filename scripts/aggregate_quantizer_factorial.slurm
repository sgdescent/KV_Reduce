#!/bin/bash
#SBATCH --job-name=kv_quant_factorial_agg
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

python3 -u aggregate_quantizer_factorial.py \
  --cell "per_token_symmetric,outputs/value_precision_sweep/qwen25_3b_15b/aggregate/summary.json,outputs/value_precision_quality/qwen25_15b/aggregate/summary.json" \
  --cell "per_token_affine,outputs/quantizer_factorial/per_token_key_affine_value/qwen25_3b_15b/spec_aggregate/summary.json,outputs/quantizer_factorial/per_token_key_affine_value/qwen25_3b_15b/quality_aggregate/summary.json" \
  --cell "per_channel_symmetric,outputs/quant_axis_ablation/qwen25_3b_15b/aggregate_affine/summary.json,outputs/quantizer_factorial/per_channel_key_symmetric_value_r0/qwen25_3b_15b/quality_aggregate/summary.json,per_channel_affine_r0" \
  --cell "per_channel_affine,outputs/kivi_objective_grid_no_residual/qwen25_3b_15b/spec_aggregate/summary.json,outputs/kivi_objective_grid_no_residual/qwen25_3b_15b/quality_aggregate/summary.json" \
  --out_dir "${OUT_DIR:-outputs/quantizer_factorial/strict_aggregate}"
