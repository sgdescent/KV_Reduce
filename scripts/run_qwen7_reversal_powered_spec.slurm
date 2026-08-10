#!/bin/bash
#SBATCH --job-name=kv_q7_rev_spec
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
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
num_prompts="${NUM_PROMPTS:-512}"
warmup_prompts="${WARMUP_PROMPTS:-2}"
shard_span=$((num_prompts + warmup_prompts))
skip_prompts=$((task_id * shard_span))
root="${SWEEP_ROOT:-outputs/kivi_reversal_powered_fineweb/qwen25_7b_3b}"
out_dir="$root/spec/ctx_1024/seed_${seed}"

export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

python3 -u benchmark_spec_kv_quantization.py \
  --big_model Qwen/Qwen2.5-7B \
  --small_model Qwen/Qwen2.5-3B \
  --big_device cuda:0 \
  --small_device cuda:1 \
  --attn_implementation sdpa \
  --dataset_name "${DATASET_NAME:-HuggingFaceFW/fineweb-edu}" \
  --dataset_config "${DATASET_CONFIG:-sample-10BT}" \
  --eval_split "${EVAL_SPLIT:-train}" \
  --stream_eval \
  --prompt_len 1024 \
  --num_prompts "$num_prompts" \
  --warmup_prompts "$warmup_prompts" \
  --skip_prompts "$skip_prompts" \
  --draft_steps 4 \
  --max_new_tokens 16 \
  --seed "$seed" \
  --quant_configs "none;k4v3;k3v4" \
  --key_quant_axis per_channel \
  --key_group_size 32 \
  --key_residual_length 128 \
  --value_quant_scheme affine \
  --out_dir "$out_dir" \
  --wandb \
  --wandb_project "${WANDB_PROJECT:-kv-reduce}" \
  --wandb_group "${WANDB_GROUP:-kivi-qwen7-reversal-powered-fineweb-spec}" \
  --wandb_run_name "qwen25_7b_3b_reversal_fineweb_spec_s${seed}"
