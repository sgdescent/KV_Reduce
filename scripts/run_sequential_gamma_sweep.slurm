#!/bin/bash
#SBATCH --job-name=kv_seq_gamma
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --exclude=catalyst-0-[9,15]
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
draft_steps_values=(2 4 8)
seed=$((task_id / ${#draft_steps_values[@]}))
draft_steps="${draft_steps_values[$((task_id % ${#draft_steps_values[@]}))]}"
num_prompts="${NUM_PROMPTS:-48}"
skip_prompts=$(( ${SKIP_BASE:-28000} + task_id * (num_prompts + 8) ))
root="${OUT_ROOT:-outputs/sequential_gamma_v1/qwen25_3b_15b}"
out_dir="$root/gamma_${draft_steps}/seed_${seed}"

export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

python3 -u benchmark_spec_kv_quantization.py \
  --big_model "${BIG_MODEL:-Qwen/Qwen2.5-3B}" \
  --small_model "${SMALL_MODEL:-Qwen/Qwen2.5-1.5B}" \
  --big_device cuda:0 \
  --small_device cuda:0 \
  --attn_implementation sdpa \
  --dataset_name HuggingFaceFW/fineweb-edu \
  --dataset_config sample-10BT \
  --eval_split train \
  --stream_eval \
  --prompt_len 1024 \
  --num_prompts "$num_prompts" \
  --warmup_prompts 0 \
  --skip_prompts "$skip_prompts" \
  --draft_steps "$draft_steps" \
  --target_verification_mode sequential \
  --max_new_tokens 16 \
  --seed "$seed" \
  --quant_configs "${QUANT_CONFIGS:-none;k8v4;k4v8;k4v4;k2v4;k4v2}" \
  --key_quant_axis per_channel \
  --key_group_size 32 \
  --key_residual_length 128 \
  --value_quant_scheme affine \
  --out_dir "$out_dir" \
  --wandb \
  --wandb_project "${WANDB_PROJECT:-kv-reduce}" \
  --wandb_group "${WANDB_GROUP:-sequential-gamma}" \
  --wandb_run_name "qwen25_3b_15b_seq_gamma${draft_steps}_s${seed}"
