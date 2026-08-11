#!/bin/bash
#SBATCH --job-name=kv_keygroup
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
group_sizes=(16 32 64 128)
group_id=$((task_id % ${#group_sizes[@]}))
seed=$((task_id / ${#group_sizes[@]}))
group_size="${group_sizes[$group_id]}"
num_prompts="${NUM_PROMPTS:-32}"
# Every group size sees the same examples within a seed. Advancing by task_id
# would confound quantizer geometry with a different text sample.
skip_prompts=$(( ${SKIP_BASE:-76000} + seed * (num_prompts + 8) ))
root="${OUT_ROOT:-outputs/key_group_geometry_v2_paired/qwen25_3b_15b}"
out_dir="$root/group_${group_size}/seed_${seed}"

export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

python3 -u benchmark_spec_kv_quantization.py \
  --big_model Qwen/Qwen2.5-3B \
  --small_model Qwen/Qwen2.5-1.5B \
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
  --draft_steps 4 \
  --target_verification_mode sequential \
  --max_new_tokens 16 \
  --seed "$seed" \
  --quant_configs "${QUANT_CONFIGS:-none;k8v4;k4v8;k4v4;k2v4;k4v2}" \
  --key_quant_axis per_channel \
  --key_group_size "$group_size" \
  --key_residual_length 128 \
  --value_quant_scheme affine \
  --out_dir "$out_dir" \
  --wandb \
  --wandb_project "${WANDB_PROJECT:-kv-reduce}" \
  --wandb_group "${WANDB_GROUP:-key-group-geometry-paired}" \
  --wandb_run_name "qwen25_3b_15b_keygroup${group_size}_s${seed}"
