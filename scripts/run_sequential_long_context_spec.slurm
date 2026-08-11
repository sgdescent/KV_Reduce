#!/bin/bash
#SBATCH --job-name=kv_seq_long
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
contexts=(4096 8192 16384)
seed=$((task_id / ${#contexts[@]}))
total_context="${contexts[$((task_id % ${#contexts[@]}))]}"
max_new_tokens=16
prompt_len=$((total_context - max_new_tokens))

case "$total_context" in
  4096) num_prompts="${NUM_PROMPTS_4K:-24}"; context_skip=0 ;;
  8192) num_prompts="${NUM_PROMPTS_8K:-12}"; context_skip=128 ;;
  16384) num_prompts="${NUM_PROMPTS_16K:-8}"; context_skip=256 ;;
  *) echo "Unsupported context length $total_context" >&2; exit 2 ;;
esac
skip_prompts=$(( ${SKIP_BASE:-0} + context_skip + seed * (num_prompts + 4) ))
root="${OUT_ROOT:-outputs/sequential_long_context_v1/qwen25_3b_15b}"
out_dir="$root/ctx_${total_context}/seed_${seed}"

export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

python3 -u benchmark_spec_kv_quantization.py \
  --big_model "${BIG_MODEL:-Qwen/Qwen2.5-3B}" \
  --small_model "${SMALL_MODEL:-Qwen/Qwen2.5-1.5B}" \
  --big_device cuda:0 \
  --small_device cuda:0 \
  --attn_implementation sdpa \
  --dataset_name emozilla/pg19 \
  --dataset_config default \
  --eval_split validation \
  --stream_eval \
  --prompt_len "$prompt_len" \
  --num_prompts "$num_prompts" \
  --warmup_prompts 0 \
  --skip_prompts "$skip_prompts" \
  --draft_steps 4 \
  --target_verification_mode sequential \
  --max_new_tokens "$max_new_tokens" \
  --seed "$seed" \
  --quant_configs "${QUANT_CONFIGS:-none;k8v4;k4v8;k4v4;k2v4;k4v2}" \
  --key_quant_axis per_channel \
  --key_group_size 32 \
  --key_residual_length 128 \
  --value_quant_scheme affine \
  --out_dir "$out_dir" \
  --wandb \
  --wandb_project "${WANDB_PROJECT:-kv-reduce}" \
  --wandb_group "${WANDB_GROUP:-sequential-long-context}" \
  --wandb_run_name "qwen25_3b_15b_seq_ctx${total_context}_s${seed}"
