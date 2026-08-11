#!/bin/bash
#SBATCH --job-name=kv_passkey_aggr
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
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
task_id=$((10#$task_id))
contexts=(8192 16384 32768)
context="${contexts[$((task_id / 3))]}"
seed=$((task_id % 3))
examples="${PASSKEY_EXAMPLES:-16}"
skip=$((seed * examples))
root="${OUT_ROOT:-outputs/kivi_passkey_aggressive_v1/qwen25_15b}"
out_dir="$root/ctx_${context}/seed_${seed}"

export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

python3 -u eval_kv_quantized_multiple_choice.py \
  --model Qwen/Qwen2.5-1.5B \
  --device cuda:0 \
  --attn_implementation sdpa \
  --task passkey \
  --num_examples "$examples" \
  --skip_examples "$skip" \
  --num_fewshot 0 \
  --max_prompt_tokens "$context" \
  --max_choice_tokens 16 \
  --seed "$seed" \
  --quant_configs "${QUANT_CONFIGS:-none;k4v4;k4v2;k2v4;k2v2}" \
  --key_quant_axis per_channel \
  --key_group_size 32 \
  --key_residual_length 128 \
  --value_quant_scheme affine \
  --out_dir "$out_dir" \
  --wandb \
  --wandb_project "${WANDB_PROJECT:-kv-reduce}" \
  --wandb_group kivi-passkey-aggressive \
  --wandb_run_name "qwen25_15b_passkey_aggressive_c${context}_s${seed}"
