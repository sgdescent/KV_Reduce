#!/bin/bash
#SBATCH --job-name=kv_fused_packed
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs outputs
source .venv/bin/activate

cmd=(
  python3 -u benchmark_fused_packed_kv_attention.py
  --contexts "${CONTEXTS:-1024;4096;16384;32768}"
  --configs "${CONFIGS:-k8v4;k4v8;k4v4}"
  --batch_size "${BATCH_SIZE:-1}"
  --query_heads "${QUERY_HEADS:-12}"
  --kv_heads "${KV_HEADS:-2}"
  --head_dim "${HEAD_DIM:-128}"
  --num_layers "${NUM_LAYERS:-28}"
  --group_size "${GROUP_SIZE:-32}"
  --residual_length "${RESIDUAL_LENGTH:-128}"
  --block_tokens "${BLOCK_TOKENS:-32}"
  --split_tokens "${SPLIT_TOKENS:-512}"
  --warmup "${WARMUP:-10}"
  --iterations "${ITERATIONS:-50}"
  --out_dir "${OUT_DIR:-outputs/fused_packed_kv_attention/qwen25_15b}"
)
if [[ "${ENABLE_WANDB:-1}" == "1" ]]; then
  cmd+=(
    --wandb
    --wandb_project "${WANDB_PROJECT:-kv-reduce}"
    --wandb_group "${WANDB_GROUP:-packed-kv-systems}"
    --wandb_run_name "${WANDB_RUN_NAME:-fused_packed_qwen25_15b}"
  )
fi

printf 'Running command:'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
