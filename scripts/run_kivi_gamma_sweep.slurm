#!/bin/bash
#SBATCH --job-name=kv_kivi_gamma
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
cd "${REPO_ROOT:-/home/sakshamg/KV_Reduce}"
mkdir -p logs outputs
if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

task_id="${SLURM_ARRAY_TASK_ID:-0}"
draft_steps_values=(2 4 8)
seeds=(0 1 2)
draft_steps="${draft_steps_values[$((task_id / ${#seeds[@]}))]}"
seed="${seeds[$((task_id % ${#seeds[@]}))]}"
root="${SWEEP_ROOT:-outputs/kivi_gamma/qwen25_3b_15b}"
out_dir="$root/gamma_${draft_steps}/seed_${seed}"
configs="${QUANT_CONFIGS:-none;k8v8;k8v4;k4v8;k4v4;k3v4;k4v3;k2v4;k4v2}"

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
  --prompt_len 1024 \
  --num_prompts "${NUM_PROMPTS:-64}" \
  --warmup_prompts 2 \
  --draft_steps "$draft_steps" \
  --max_new_tokens 16 \
  --seed "$seed" \
  --shuffle_eval \
  --quant_configs "$configs" \
  --key_quant_axis per_channel \
  --key_group_size 32 \
  --key_residual_length 128 \
  --value_quant_scheme affine \
  --out_dir "$out_dir" \
  --wandb \
  --wandb_project "${WANDB_PROJECT:-kv-reduce}" \
  --wandb_group "${WANDB_GROUP:-kivi-gamma}" \
  --wandb_run_name "kivi_gamma_g${draft_steps}_s${seed}"
