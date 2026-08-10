#!/bin/bash
#SBATCH --job-name=kv_key_axis
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
repo_root="${REPO_ROOT:-/home/sakshamg/KV_Reduce}"
cd "$repo_root"
mkdir -p logs outputs
if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

task_id="${SLURM_ARRAY_TASK_ID:-0}"
contexts=(1024 4096)
seeds=(0 1 2)
context="${contexts[$((task_id / ${#seeds[@]}))]}"
seed="${seeds[$((task_id % ${#seeds[@]}))]}"
num_prompts="${NUM_PROMPTS_1K:-64}"
if (( context == 4096 )); then
  num_prompts="${NUM_PROMPTS_4K:-32}"
fi

axis="${KEY_QUANT_AXIS:-per_channel}"
group_size="${KEY_GROUP_SIZE:-32}"
residual_length="${KEY_RESIDUAL_LENGTH:-128}"
variant="${AXIS_VARIANT:-${axis}_r${residual_length}}"
root="${SWEEP_ROOT:-outputs/quant_axis_ablation/qwen25_3b_15b/$variant}"
out_dir="$root/ctx_${context}/seed_${seed}"

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
  --quant_configs "${QUANT_CONFIGS:-none;k8v8;k8v4;k4v8;k4v4}" \
  --key_quant_axis "$axis" \
  --key_group_size "$group_size" \
  --key_residual_length "$residual_length" \
  --out_dir "$out_dir" \
  --wandb \
  --wandb_project "${WANDB_PROJECT:-kv-reduce}" \
  --wandb_group "${WANDB_GROUP:-key-quant-axis}" \
  --wandb_run_name "key_axis_${variant}_c${context}_s${seed}"
