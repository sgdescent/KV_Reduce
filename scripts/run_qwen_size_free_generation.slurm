#!/bin/bash
#SBATCH --job-name=kv_qwen_gen
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
model_id=$((task_id % 2))
generation_id=$(((task_id / 2) % 2))
seed=$((task_id / 4))
case "$model_id" in
  0) tag=qwen25_3b; model=Qwen/Qwen2.5-3B ;;
  1) tag=qwen3_4b; model=Qwen/Qwen3-4B ;;
  *) echo "Unsupported model id $model_id" >&2; exit 2 ;;
esac
case "$generation_id" in
  0) max_new_tokens=64; num_prompts="${NUM_PROMPTS_64:-32}" ;;
  1) max_new_tokens=256; num_prompts="${NUM_PROMPTS_256:-16}" ;;
  *) echo "Unsupported generation id $generation_id" >&2; exit 2 ;;
esac

skip_prompts=$(( ${SKIP_BASE:-40000} + task_id * 40 ))
root="${OUT_ROOT:-outputs/kivi_free_generation_v1_retry/qwen_size_sweep}"
out_dir="$root/$tag/ctx_1024/gen_${max_new_tokens}/seed_${seed}"

export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

python3 -u eval_kv_quantized_generation.py \
  --model "$model" \
  --device cuda:0 \
  --dtype bf16 \
  --dataset_name HuggingFaceFW/fineweb-edu \
  --dataset_config sample-10BT \
  --split train \
  --streaming \
  --skip_prompts "$skip_prompts" \
  --num_prompts "$num_prompts" \
  --prompt_len 1024 \
  --max_new_tokens "$max_new_tokens" \
  --quant_configs "${QUANT_CONFIGS:-none;k8v4;k4v8;k4v4}" \
  --key_quant_axis per_channel \
  --key_group_size 32 \
  --key_residual_length 128 \
  --value_quant_scheme affine \
  --seed "$seed" \
  --out_dir "$out_dir" \
  --wandb \
  --wandb_project "${WANDB_PROJECT:-kv-reduce}" \
  --wandb_group "${WANDB_GROUP:-qwen-size-free-generation}" \
  --wandb_run_name "${tag}_free_gen${max_new_tokens}_s${seed}"
