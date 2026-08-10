#!/bin/bash
#SBATCH --job-name=kv_kivi_xagg
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
root="${ROOT:-outputs/kivi_objective_cross_family}"

python3 -u aggregate_value_precision_sweep.py \
  --sweep_dir "$root/spec/$pair" \
  --out_dir "$root/spec/$pair/aggregate"
python3 -u aggregate_value_precision_quality_sweep.py \
  --sweep_dir "$root/quality/$pair" \
  --out_dir "$root/quality/$pair/aggregate"
python3 -u compare_value_precision_objectives.py \
  --spec_summary "$root/spec/$pair/aggregate/summary.json" \
  --quality_summary "$root/quality/$pair/aggregate/summary.json" \
  --out_dir "$root/comparison/$pair"
python3 -u aggregate_kivi_objective_preferences.py \
  --spec_dir "$root/spec/$pair" \
  --quality_dir "$root/quality/$pair" \
  --config_pairs "k8v4,k4v8;k4v3,k3v4;k4v2,k2v4" \
  --out_dir "$root/comparison/$pair"
