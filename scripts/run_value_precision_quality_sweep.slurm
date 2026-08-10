#!/bin/bash
#SBATCH --job-name=kv_value_quality
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
contexts=(1024 4096 16384)
seeds=(0 1 2)
context="${contexts[$((task_id / ${#seeds[@]}))]}"
seed="${seeds[$((task_id % ${#seeds[@]}))]}"

dataset_name=wikitext
dataset_config=wikitext-2-raw-v1
eval_split=validation
num_sequences="${NUM_SEQUENCES_1K:-64}"
stream_args=()
if (( context == 4096 )); then
  num_sequences="${NUM_SEQUENCES_4K:-32}"
elif (( context == 16384 )); then
  dataset_name=emozilla/pg19
  dataset_config=default
  num_sequences="${NUM_SEQUENCES_16K:-8}"
  stream_args+=(--stream_eval)
fi

out_dir="${SWEEP_ROOT:-outputs/value_precision_quality/qwen25_15b}/ctx_${context}/seed_${seed}"
quant_configs="${QUANT_CONFIGS:-none;k16v8;k16v4;k16v3;k16v2;k8v8;k8v4;k8v3;k8v2;k4v8;k4v4}"

python3 -u profile_kv_quality_sensitivity.py \
  --model "${SMALL_MODEL:-Qwen/Qwen2.5-1.5B}" \
  --device cuda:0 \
  --attn_implementation sdpa \
  --dataset_name "$dataset_name" \
  --dataset_config "$dataset_config" \
  --eval_split "$eval_split" \
  --prompt_len "$context" \
  --continuation_len "${CONTINUATION_LEN:-32}" \
  --num_sequences "$num_sequences" \
  --seed "$seed" \
  --shuffle_eval \
  --quant_configs "$quant_configs" \
  --out_dir "$out_dir" \
  --wandb \
  --wandb_project "${WANDB_PROJECT:-kv-reduce}" \
  --wandb_group "${WANDB_GROUP:-value-precision-quality}" \
  --wandb_run_name "value_quality_c${context}_s${seed}" \
  "${stream_args[@]}"
