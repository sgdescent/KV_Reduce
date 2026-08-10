#!/bin/bash
#SBATCH --job-name=kv_key_axis_agg
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
repo_root="${REPO_ROOT:-/home/sakshamg/KV_Reduce}"
cd "$repo_root"
if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

python3 -u aggregate_quant_axis_ablation.py \
  --per_token_dir "${PER_TOKEN_DIR:-outputs/value_precision_sweep/qwen25_3b_15b}" \
  --variant "per_channel_affine_r0=${R0_DIR:-outputs/quant_axis_ablation/qwen25_3b_15b/per_channel_affine_r0}" \
  --variant "per_channel_affine_r128=${R128_DIR:-outputs/quant_axis_ablation/qwen25_3b_15b/per_channel_affine_r128}" \
  --out_dir "${OUT_DIR:-outputs/quant_axis_ablation/qwen25_3b_15b/aggregate_affine}"
