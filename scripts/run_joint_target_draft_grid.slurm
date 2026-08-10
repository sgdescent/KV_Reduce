#!/bin/bash
#SBATCH --job-name=kv_joint_grid
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
cd "${REPO_ROOT:-/home/sakshamg/KV_Reduce}"
mkdir -p logs outputs
if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

task_id="${SLURM_ARRAY_TASK_ID:-0}"
contexts=(1024 4096)
seeds=(40 41 42)
context="${contexts[$((task_id / ${#seeds[@]}))]}"
seed="${seeds[$((task_id % ${#seeds[@]}))]}"
num_prompts="${NUM_PROMPTS_1K:-64}"
if (( context == 4096 )); then
  num_prompts="${NUM_PROMPTS_4K:-32}"
fi

root="${ROOT:-outputs/kivi_joint_target_draft/qwen25_3b_15b}"
out_dir="$root/ctx_${context}/seed_${seed}"
target_configs="${TARGET_QUANT_CONFIGS:-none;k8v8;k4v8;k8v4;k4v4}"
draft_configs="${QUANT_CONFIGS:-none;k8v8;k4v8;k8v4;k4v4}"

export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

python3 -u benchmark_spec_kv_quantization.py \
  --big_model "${BIG_MODEL:-Qwen/Qwen2.5-3B}" \
  --small_model "${SMALL_MODEL:-Qwen/Qwen2.5-1.5B}" \
  --big_device cuda:0 \
  --small_device cuda:0 \
  --attn_implementation sdpa \
  --dataset_name wikitext \
  --dataset_config wikitext-2-raw-v1 \
  --eval_split validation \
  --prompt_len "$context" \
  --num_prompts "$num_prompts" \
  --warmup_prompts 2 \
  --draft_steps 4 \
  --max_new_tokens 16 \
  --seed "$seed" \
  --shuffle_eval \
  --target_quant_configs "$target_configs" \
  --quant_configs "$draft_configs" \
  --key_quant_axis per_channel \
  --key_group_size 32 \
  --key_residual_length 128 \
  --value_quant_scheme affine \
  --out_dir "$out_dir" \
  --wandb \
  --wandb_project "${WANDB_PROJECT:-kv-reduce}" \
  --wandb_group "${WANDB_GROUP:-kivi-joint-target-draft}" \
  --wandb_run_name "kivi_joint_c${context}_s${seed}"
