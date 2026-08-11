#!/bin/bash
#SBATCH --job-name=kv_seq_xspec2
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=10:00:00
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
pair_id=$((task_id % 3))
seed=$((task_id / 3))
case "$pair_id" in
  0) pair=qwen25_7b_3b; big=Qwen/Qwen2.5-7B; small=Qwen/Qwen2.5-3B ;;
  1) pair=qwen3_8b_4b; big=Qwen/Qwen3-8B; small=Qwen/Qwen3-4B ;;
  2) pair=llama31_8b_llama32_3b; big=meta-llama/Llama-3.1-8B; small=meta-llama/Llama-3.2-3B ;;
  *) echo "Unsupported pair id $pair_id" >&2; exit 2 ;;
esac

num_prompts="${NUM_PROMPTS:-32}"
skip_prompts=$(( ${SKIP_BASE:-24000} + task_id * (num_prompts + 8) ))
root="${OUT_ROOT:-outputs/sequential_cross_family_v1}"
out_dir="$root/$pair/ctx_1024/seed_${seed}"

export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

python3 -u benchmark_spec_kv_quantization.py \
  --big_model "$big" \
  --small_model "$small" \
  --big_device cuda:0 \
  --small_device cuda:1 \
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
  --quant_configs "${QUANT_CONFIGS:-none;k8v4;k4v8;k4v4}" \
  --key_quant_axis per_channel \
  --key_group_size 32 \
  --key_residual_length 128 \
  --value_quant_scheme affine \
  --out_dir "$out_dir" \
  --wandb \
  --wandb_project "${WANDB_PROJECT:-kv-reduce}" \
  --wandb_group "${WANDB_GROUP:-sequential-cross-family}" \
  --wandb_run_name "${pair}_sequential_s${seed}"
