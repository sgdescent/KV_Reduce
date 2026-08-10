#!/bin/bash
#SBATCH --job-name=kv_kivi_mc
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
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
num_seeds="${NUM_SEEDS:-3}"
if (( num_seeds <= 0 )); then
  echo "NUM_SEEDS must be positive" >&2
  exit 2
fi
task_group=$((task_id / num_seeds))
seed=$((task_id % num_seeds))
case "$task_group" in
  0)
    task=hellaswag
    examples="${HELLASWAG_EXAMPLES:-256}"
    ;;
  1)
    task=arc_challenge
    examples="${ARC_EXAMPLES:-256}"
    ;;
  *)
    echo "Unsupported task group $task_group" >&2
    exit 2
    ;;
esac

model="${MODEL:-Qwen/Qwen2.5-1.5B}"
model_label="${MODEL_LABEL:-qwen25_15b}"
root="${OUT_ROOT:-outputs/kivi_multiple_choice/$model_label}"
out_dir="$root/$task/seed_$seed"
skip=$((seed * examples))
configs="${QUANT_CONFIGS:-none;k8v4;k4v8;k4v4;k3v4;k4v3}"

export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

python3 -u eval_kv_quantized_multiple_choice.py \
  --model "$model" \
  --device cuda:0 \
  --attn_implementation sdpa \
  --task "$task" \
  --num_examples "$examples" \
  --skip_examples "$skip" \
  --seed "$seed" \
  --quant_configs "$configs" \
  --key_quant_axis per_channel \
  --key_group_size 32 \
  --key_residual_length 128 \
  --value_quant_scheme affine \
  --out_dir "$out_dir" \
  --wandb \
  --wandb_project "${WANDB_PROJECT:-kv-reduce}" \
  --wandb_group "${WANDB_GROUP:-kivi-multiple-choice}" \
  --wandb_run_name "${model_label}_${task}_kivi_s${seed}"
