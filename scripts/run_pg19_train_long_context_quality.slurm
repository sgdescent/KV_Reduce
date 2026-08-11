#!/bin/bash
#SBATCH --job-name=kv_qual_pg19
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
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
contexts=(16384 32768)
seed=$((task_id / ${#contexts[@]}))
total_context="${contexts[$((task_id % ${#contexts[@]}))]}"
continuation_len=16
prompt_len=$((total_context - continuation_len))
case "$total_context" in
  16384) num_sequences="${NUM_SEQUENCES_16K:-12}"; context_skip=1024 ;;
  32768) num_sequences="${NUM_SEQUENCES_32K:-8}"; context_skip=2048 ;;
  *) echo "Unsupported context length $total_context" >&2; exit 2 ;;
esac
skip_sequences=$(( ${SKIP_BASE:-0} + context_skip + seed * (num_sequences + 4) ))
root="${OUT_ROOT:-outputs/quality_long_context_v2_pg19_train/qwen25_15b}"
out_dir="$root/ctx_${total_context}/seed_${seed}"

export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

python3 -u profile_kv_quality_sensitivity.py \
  --model "${MODEL:-Qwen/Qwen2.5-1.5B}" \
  --device cuda:0 \
  --attn_implementation sdpa \
  --dataset_name emozilla/pg19 \
  --dataset_config default \
  --eval_split train \
  --stream_eval \
  --prompt_len "$prompt_len" \
  --continuation_len "$continuation_len" \
  --num_sequences "$num_sequences" \
  --skip_sequences "$skip_sequences" \
  --seed "$seed" \
  --quant_configs "${QUANT_CONFIGS:-none;k8v4;k4v8;k4v4;k2v4;k4v2}" \
  --key_quant_axis per_channel \
  --key_group_size 32 \
  --key_residual_length 128 \
  --value_quant_scheme affine \
  --out_dir "$out_dir" \
  --wandb \
  --wandb_project "${WANDB_PROJECT:-kv-reduce}" \
  --wandb_group "${WANDB_GROUP:-quality-pg19-train-long-context}" \
  --wandb_run_name "qwen25_15b_quality_pg19_train_ctx${total_context}_s${seed}"
