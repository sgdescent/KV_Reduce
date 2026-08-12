#!/bin/bash
#SBATCH --job-name=kv_passkey_lsel
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --exclude=catalyst-0-[9,15]
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
cd "${REPO_ROOT:-/home/sakshamg/KV_Reduce}"
if [[ -f .venv/bin/activate ]]; then source .venv/bin/activate; fi
task_id=$((10#${SLURM_ARRAY_TASK_ID:-0}))
case "$task_id" in
  0) model_tag="qwen25_15b" ;;
  1) model_tag="llama32_3b" ;;
  2) model_tag="qwen3_4b" ;;
  *) exit 2 ;;
esac
profile_root="${PROFILE_ROOT:-outputs/passkey_layer_sensitivity_v1}/$model_tag"
out_dir="${OUT_ROOT:-outputs/passkey_layer_policy_v1}/$model_tag/allocations"
python3 -u select_passkey_layer_policy.py \
  --sensitivity_summary "$profile_root/aggregate/summary.json" \
  --sensitivity_manifest "$profile_root/allocations/manifest.json" \
  --num_reductions "${NUM_REDUCTIONS:-8}" \
  --out_dir "$out_dir"
