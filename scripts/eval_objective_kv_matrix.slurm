#!/bin/bash
#SBATCH --job-name=kv_obj_matrix
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
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs outputs
if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

MANIFEST="${MANIFEST:?Set MANIFEST}"
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "$MANIFEST")
if [[ -z "$line" ]]; then
  echo "No manifest row for task $SLURM_ARRAY_TASK_ID" >&2
  exit 1
fi
IFS=$'\t' read -r objective budget context seed num_eval quant_configs quality_allocation acceptance_allocation out_dir <<< "$line"
mkdir -p "$out_dir"

common_wandb=()
if [[ "${ENABLE_WANDB:-1}" == "1" ]]; then
  common_wandb=(--wandb --wandb_project "${WANDB_PROJECT:-kv-reduce}" --wandb_group "${WANDB_GROUP:-objective-matrix}")
fi
stream_args=()
if [[ "${STREAM_EVAL:-0}" == "1" ]]; then
  stream_args+=(--stream_eval)
fi
if [[ "$objective" == "quality" ]]; then
  python3 -u profile_kv_quality_sensitivity.py \
    --model "${SMALL_MODEL:-Qwen/Qwen2.5-1.5B}" \
    --device cuda:0 \
    --dataset_name "${DATASET_NAME:-wikitext}" \
    --dataset_config "${DATASET_CONFIG:-wikitext-2-raw-v1}" \
    --eval_split "${EVAL_SPLIT:-validation}" \
    --prompt_len "$context" \
    --continuation_len "${CONTINUATION_LEN:-32}" \
    --num_sequences "$num_eval" \
    --seed "$seed" \
    --shuffle_eval \
    --quant_configs "$quant_configs" \
    --out_dir "$out_dir" \
    --wandb_run_name "quality_b${budget}_c${context}_s${seed}" \
    "${stream_args[@]}" \
    "${common_wandb[@]}"
elif [[ "$objective" == "acceptance" ]]; then
  python3 -u benchmark_spec_kv_quantization.py \
    --big_model "${BIG_MODEL:-Qwen/Qwen2.5-3B}" \
    --small_model "${SMALL_MODEL:-Qwen/Qwen2.5-1.5B}" \
    --big_device cuda:0 \
    --small_device cuda:0 \
    --dataset_name "${DATASET_NAME:-wikitext}" \
    --dataset_config "${DATASET_CONFIG:-wikitext-2-raw-v1}" \
    --eval_split "${EVAL_SPLIT:-validation}" \
    --prompt_len "$context" \
    --num_prompts "$num_eval" \
    --warmup_prompts 2 \
    --draft_steps "${DRAFT_STEPS:-4}" \
    --max_new_tokens "${MAX_NEW_TOKENS:-16}" \
    --seed "$seed" \
    --shuffle_eval \
    --quant_configs "$quant_configs" \
    --out_dir "$out_dir" \
    --wandb_run_name "acceptance_b${budget}_c${context}_s${seed}" \
    "${stream_args[@]}" \
    "${common_wandb[@]}"
else
  echo "Unknown objective: $objective" >&2
  exit 1
fi
