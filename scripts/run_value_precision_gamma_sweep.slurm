#!/bin/bash
#SBATCH --job-name=kv_value_gamma
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
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
root="${SWEEP_ROOT:-outputs/value_precision_gamma/qwen25_3b_15b}"
out_dir="$root/gamma_${draft_steps}/seed_${seed}"
quant_configs="${QUANT_CONFIGS:-none;k8v8;k8v4;k8v3;k8v2;k4v8}"

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
  --quant_configs "$quant_configs" \
  --out_dir "$out_dir" \
  --wandb \
  --wandb_project "${WANDB_PROJECT:-kv-reduce}" \
  --wandb_group "${WANDB_GROUP:-value-precision-gamma}" \
  --wandb_run_name "value_gamma_g${draft_steps}_s${seed}"
