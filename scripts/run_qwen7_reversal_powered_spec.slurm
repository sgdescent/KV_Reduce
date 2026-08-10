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
seed="${seeds[${SLURM_ARRAY_TASK_ID:-0}]}"
root="${SWEEP_ROOT:-outputs/kivi_reversal_powered/qwen25_7b_3b}"
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
  --dataset_name wikitext \
  --dataset_config wikitext-2-raw-v1 \
  --eval_split validation \
  --prompt_len 1024 \
  --num_prompts "${NUM_PROMPTS:-512}" \
  --warmup_prompts 2 \
  --draft_steps 4 \
  --max_new_tokens 16 \
  --seed "$seed" \
  --shuffle_eval \
  --quant_configs "none;k4v3;k3v4" \
  --key_quant_axis per_channel \
  --key_group_size 32 \
  --key_residual_length 128 \
  --value_quant_scheme affine \
  --out_dir "$out_dir" \
  --wandb \
  --wandb_project "${WANDB_PROJECT:-kv-reduce}" \
  --wandb_group "${WANDB_GROUP:-kivi-qwen7-reversal-powered-spec}" \
  --wandb_run_name "qwen25_7b_3b_reversal_spec_s${seed}"
