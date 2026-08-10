#!/bin/bash
#SBATCH --job-name=kv_q7_rev_qual
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
cd "${REPO_ROOT:-/home/sakshamg/KV_Reduce}"
mkdir -p logs outputs
if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

seeds=(30 31 32)
task_id="${SLURM_ARRAY_TASK_ID:-0}"
seed="${seeds[$task_id]}"
num_sequences="${NUM_SEQUENCES:-256}"
skip_sequences=$((task_id * num_sequences))
root="${SWEEP_ROOT:-outputs/kivi_reversal_powered_fineweb/qwen25_7b_3b}"
out_dir="$root/quality/ctx_1024/seed_${seed}"

export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

python3 -u profile_kv_quality_sensitivity.py \
  --model Qwen/Qwen2.5-3B \
  --device cuda:0 \
  --attn_implementation sdpa \
  --dataset_name "${DATASET_NAME:-HuggingFaceFW/fineweb-edu}" \
  --dataset_config "${DATASET_CONFIG:-sample-10BT}" \
  --eval_split "${EVAL_SPLIT:-train}" \
  --stream_eval \
  --prompt_len 1024 \
  --continuation_len 32 \
  --num_sequences "$num_sequences" \
  --skip_sequences "$skip_sequences" \
  --seed "$seed" \
  --quant_configs "none;k4v3;k3v4" \
  --key_quant_axis per_channel \
  --key_group_size 32 \
  --key_residual_length 128 \
  --value_quant_scheme affine \
  --out_dir "$out_dir" \
  --wandb \
  --wandb_project "${WANDB_PROJECT:-kv-reduce}" \
  --wandb_group "${WANDB_GROUP:-kivi-qwen7-reversal-powered-fineweb-quality}" \
  --wandb_run_name "qwen25_7b_3b_reversal_fineweb_quality_s${seed}"
