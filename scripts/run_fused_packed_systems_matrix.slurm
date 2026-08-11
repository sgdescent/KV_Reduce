#!/bin/bash
#SBATCH --job-name=kv_pack_matrix
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
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
shape_id=$((task_id % 6))
batch_id=$((task_id / 6))
batches=(1 4 16)
batch_size="${batches[$batch_id]}"

case "$shape_id" in
  0) tag=qwen25_15b; query_heads=12; kv_heads=2; head_dim=128; num_layers=28; contexts="1024;4096;16384;32768" ;;
  1) tag=qwen25_3b; query_heads=16; kv_heads=2; head_dim=128; num_layers=36; contexts="1024;4096;16384;32768" ;;
  2) tag=qwen3_4b; query_heads=32; kv_heads=8; head_dim=80; num_layers=36; contexts="1024;4096;16384;32768" ;;
  3) tag=llama32_3b; query_heads=24; kv_heads=8; head_dim=128; num_layers=28; contexts="1024;4096;16384;32768" ;;
  4) tag=olmo2_1b; query_heads=16; kv_heads=16; head_dim=128; num_layers=16; contexts="1024;4096" ;;
  5) tag=smollm2_360m; query_heads=15; kv_heads=5; head_dim=64; num_layers=32; contexts="1024;4096;8192" ;;
  *) echo "Unsupported shape id $shape_id" >&2; exit 2 ;;
esac

root="${OUT_ROOT:-outputs/fused_packed_systems_matrix_v1}"
out_dir="$root/$tag/batch_${batch_size}"

python3 -u benchmark_fused_packed_kv_attention.py \
  --contexts "$contexts" \
  --configs "${CONFIGS:-k8v4;k4v8;k4v4}" \
  --batch_size "$batch_size" \
  --query_heads "$query_heads" \
  --kv_heads "$kv_heads" \
  --head_dim "$head_dim" \
  --num_layers "$num_layers" \
  --group_size 32 \
  --residual_length 128 \
  --block_tokens_list "${BLOCK_TOKENS_LIST:-32;64}" \
  --split_tokens_list "${SPLIT_TOKENS_LIST:-0;256;512;1024}" \
  --warmup "${WARMUP:-5}" \
  --iterations "${ITERATIONS:-30}" \
  --seed "$((1729 + task_id))" \
  --out_dir "$out_dir" \
  --wandb \
  --wandb_project "${WANDB_PROJECT:-kv-reduce}" \
  --wandb_group "${WANDB_GROUP:-packed-kv-systems-matrix}" \
  --wandb_run_name "packed_${tag}_b${batch_size}"
