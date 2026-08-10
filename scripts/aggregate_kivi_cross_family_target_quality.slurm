#!/bin/bash
#SBATCH --job-name=kv_kivi_xtqagg
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
cd "${REPO_ROOT:-/home/sakshamg/KV_Reduce}"
if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

pairs=(qwen25_3b_15b qwen25_7b_3b qwen3_8b_4b llama31_8b_llama32_3b olmo2_7b_1b smollm2_17b_360m)
pair="${pairs[${SLURM_ARRAY_TASK_ID:-0}]}"
root="${TARGET_QUALITY_ROOT:-outputs/kivi_objective_cross_family/target_quality}"

python3 -u aggregate_value_precision_quality_sweep.py \
  --sweep_dir "$root/$pair" \
  --out_dir "$root/$pair/aggregate"
