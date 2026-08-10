#!/bin/bash
#SBATCH --job-name=kv_value_q_xfam
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
pair_id=$((task_id % 6))
seed="${VALUE_SEED:-$((20 + task_id / 6))}"

case "$pair_id" in
  0)
    pair=qwen25_3b_15b
    model=Qwen/Qwen2.5-1.5B
    ;;
  1)
    pair=qwen25_7b_3b
    model=Qwen/Qwen2.5-3B
    ;;
  2)
    pair=qwen3_8b_4b
    model=Qwen/Qwen3-4B
    ;;
  3)
    pair=llama31_8b_llama32_3b
    model=meta-llama/Llama-3.2-3B
    ;;
  4)
    pair=olmo2_7b_1b
    model=allenai/OLMo-2-0425-1B
    ;;
  5)
    pair=smollm2_17b_360m
    model=HuggingFaceTB/SmolLM2-360M
    ;;
  *)
    echo "Unsupported pair id $pair_id" >&2
    exit 2
    ;;
esac

root="${QUALITY_SWEEP_ROOT:-outputs/value_precision_quality_cross_family}"
out_dir="$root/$pair/ctx_1024/seed_${seed}"
quant_configs="${QUANT_CONFIGS:-none;k8v8;k8v4;k8v3;k8v2;k4v8}"

export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

python3 -u profile_kv_quality_sensitivity.py \
  --model "$model" \
  --device cuda:0 \
  --attn_implementation sdpa \
  --dataset_name wikitext \
  --dataset_config wikitext-2-raw-v1 \
  --eval_split validation \
  --prompt_len 1024 \
  --continuation_len "${CONTINUATION_LEN:-32}" \
  --num_sequences "${NUM_SEQUENCES:-32}" \
  --seed "$seed" \
  --shuffle_eval \
  --quant_configs "$quant_configs" \
  --out_dir "$out_dir" \
  --wandb \
  --wandb_project "${WANDB_PROJECT:-kv-reduce}" \
  --wandb_group "${WANDB_GROUP:-value-precision-quality-cross-family}" \
  --wandb_run_name "${pair}_value_quality_c1024_s${seed}"
