#!/bin/bash
#SBATCH --job-name=kv_packed_attn
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs outputs
if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

cmd=(
  python3 -u benchmark_packed_kv_attention.py
  --device cuda:0
  --dtype "${DTYPE:-bf16}"
  --contexts "${CONTEXTS:-1024;4096;16384;32768}"
  --configs "${QUANT_CONFIGS:-none;k8v4;k4v8;k4v4;k3v4;k4v3}"
  --batch_size "${BATCH_SIZE:-1}"
  --query_heads "${QUERY_HEADS:-12}"
  --kv_heads "${KV_HEADS:-2}"
  --head_dim "${HEAD_DIM:-128}"
  --num_layers "${NUM_LAYERS:-28}"
  --group_size "${KEY_GROUP_SIZE:-32}"
  --residual_length "${KEY_RESIDUAL_LENGTH:-128}"
  --warmup "${WARMUP:-5}"
  --iterations "${ITERATIONS:-20}"
  --out_dir "${OUT_DIR:-outputs/packed_kv_attention_${SLURM_JOB_ID:-local}}"
)
if [[ "${ENABLE_WANDB:-1}" == "1" ]]; then
  cmd+=(
    --wandb
    --wandb_project "${WANDB_PROJECT:-kv-reduce}"
    --wandb_group "${WANDB_GROUP:-packed-kv-systems}"
    --wandb_run_name "${WANDB_RUN_NAME:-packed_kv_attention_${SLURM_JOB_ID:-local}}"
  )
fi

printf 'Running command:'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
