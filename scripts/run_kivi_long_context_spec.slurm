#!/bin/bash
#SBATCH --job-name=kv_kivi_lspec
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
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
total_contexts=(16384 32768)
seeds=(0 1 2)
total_context="${total_contexts[$((task_id / ${#seeds[@]}))]}"
seed="${seeds[$((task_id % ${#seeds[@]}))]}"
num_prompts="${NUM_PROMPTS_16K:-8}"
max_new_tokens=16
prompt_len="$total_context"
if (( total_context == 32768 )); then
  num_prompts="${NUM_PROMPTS_32K:-4}"
  prompt_len=$((total_context - max_new_tokens))
fi

root="${SPEC_ROOT:-outputs/kivi_objective_long_context/qwen25_3b_15b/spec}"
out_dir="$root/ctx_${total_context}/seed_${seed}"
configs="${QUANT_CONFIGS:-none;k8v4;k4v8;k4v4;k3v3;k2v4;k4v2;k2v2}"

export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

python3 -u benchmark_spec_kv_quantization.py \
  --big_model Qwen/Qwen2.5-3B \
  --small_model Qwen/Qwen2.5-1.5B \
  --big_device cuda:0 \
  --small_device cuda:0 \
  --attn_implementation sdpa \
  --dataset_name emozilla/pg19 \
  --dataset_config default \
  --eval_split validation \
  --stream_eval \
  --shuffle_eval \
  --prompt_len "$prompt_len" \
  --num_prompts "$num_prompts" \
  --warmup_prompts 1 \
  --draft_steps 4 \
  --max_new_tokens "$max_new_tokens" \
  --seed "$seed" \
  --quant_configs "$configs" \
  --key_quant_axis per_channel \
  --key_group_size 32 \
  --key_residual_length 128 \
  --value_quant_scheme affine \
  --out_dir "$out_dir" \
  --wandb \
  --wandb_project "${WANDB_PROJECT:-kv-reduce}" \
  --wandb_group kivi-objective-long-context-spec \
  --wandb_run_name "kivi_long_spec_c${total_context}_s${seed}"
