#!/bin/bash
#SBATCH --job-name=kv_target_quant
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/kv_target_quant-%j.out
#SBATCH --error=logs/kv_target_quant-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

if [ -d ".venv" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

MODEL="${MODEL:-Qwen/Qwen2.5-3B}"
DEVICE="${DEVICE:-cuda:0}"
DATASET_NAME="${DATASET_NAME:-wikitext}"
DATASET_CONFIG="${DATASET_CONFIG:-wikitext-2-raw-v1}"
SPLIT="${SPLIT:-test}"
NUM_SEQUENCES="${NUM_SEQUENCES:-128}"
PROMPT_LEN="${PROMPT_LEN:-1024}"
BITS="${BITS:-8,4,3}"
OUT_DIR="${OUT_DIR:-outputs/target_cache_quant_eval}"
ENABLE_WANDB="${ENABLE_WANDB:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-kv-reduce}"
WANDB_GROUP="${WANDB_GROUP:-target-cache-quant}"

cmd=(
  python3 -u eval_target_cache_quantization.py
  --model "$MODEL"
  --device "$DEVICE"
  --dataset_name "$DATASET_NAME"
  --dataset_config "$DATASET_CONFIG"
  --split "$SPLIT"
  --num_sequences "$NUM_SEQUENCES"
  --prompt_len "$PROMPT_LEN"
  --bits "$BITS"
  --out_dir "$OUT_DIR"
)

if [ "$ENABLE_WANDB" = "1" ]; then
  cmd+=(--wandb --wandb_project "$WANDB_PROJECT" --wandb_group "$WANDB_GROUP")
fi

printf 'Running command:\n '
printf '%q ' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
