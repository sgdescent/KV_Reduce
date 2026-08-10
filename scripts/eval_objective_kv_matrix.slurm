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
IFS=$'\t' read -r -a manifest_fields <<< "$line"
if (( ${#manifest_fields[@]} == 9 )); then
  # Original matrix schema without matched-memory heuristic baselines.
  objective="${manifest_fields[0]}"
  budget="${manifest_fields[1]}"
  context="${manifest_fields[2]}"
  seed="${manifest_fields[3]}"
  num_eval="${manifest_fields[4]}"
  quant_configs="${manifest_fields[5]}"
  quality_allocation="${manifest_fields[6]}"
  acceptance_allocation="${manifest_fields[7]}"
  out_dir="${manifest_fields[8]}"
  skip_blocks=0
  warmup_prompts=2
  shuffle_args=(--shuffle_eval)
elif (( ${#manifest_fields[@]} == 11 )); then
  # Extended schema adds K-priority and V-priority allocation paths.
  objective="${manifest_fields[0]}"
  budget="${manifest_fields[1]}"
  context="${manifest_fields[2]}"
  seed="${manifest_fields[3]}"
  num_eval="${manifest_fields[4]}"
  quant_configs="${manifest_fields[5]}"
  quality_allocation="${manifest_fields[6]}"
  acceptance_allocation="${manifest_fields[7]}"
  out_dir="${manifest_fields[10]}"
  skip_blocks=0
  warmup_prompts=2
  shuffle_args=(--shuffle_eval)
elif (( ${#manifest_fields[@]} == 13 )); then
  # Held-out schema adds an explicit disjoint token-block offset.
  objective="${manifest_fields[0]}"
  budget="${manifest_fields[1]}"
  context="${manifest_fields[2]}"
  seed="${manifest_fields[3]}"
  num_eval="${manifest_fields[4]}"
  skip_blocks="${manifest_fields[5]}"
  warmup_prompts="${manifest_fields[6]}"
  quant_configs="${manifest_fields[7]}"
  quality_allocation="${manifest_fields[8]}"
  acceptance_allocation="${manifest_fields[9]}"
  out_dir="${manifest_fields[12]}"
  # Explicit offsets are disjoint only when every shard traverses the same
  # dataset order. The seed remains available for model-side randomness.
  shuffle_args=()
else
  echo "Unsupported manifest schema with ${#manifest_fields[@]} fields" >&2
  exit 1
fi
mkdir -p "$out_dir"

common_wandb=()
if [[ "${ENABLE_WANDB:-1}" == "1" ]]; then
  common_wandb=(--wandb --wandb_project "${WANDB_PROJECT:-kv-reduce}" --wandb_group "${WANDB_GROUP:-objective-matrix}")
fi
stream_args=()
if [[ "${STREAM_EVAL:-0}" == "1" ]]; then
  stream_args+=(--stream_eval)
fi
text_column_args=()
if [[ -n "${TEXT_COLUMN:-}" ]]; then
  text_column_args+=(--text_column "$TEXT_COLUMN")
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
    --skip_sequences "$skip_blocks" \
    --seed "$seed" \
    "${shuffle_args[@]}" \
    --quant_configs "$quant_configs" \
    --out_dir "$out_dir" \
    --wandb_run_name "quality_b${budget}_c${context}_s${seed}" \
    "${stream_args[@]}" \
    "${text_column_args[@]}" \
    "${common_wandb[@]}"
elif [[ "$objective" == "acceptance" ]]; then
  python3 -u benchmark_spec_kv_quantization.py \
    --big_model "${BIG_MODEL:-Qwen/Qwen2.5-3B}" \
    --small_model "${SMALL_MODEL:-Qwen/Qwen2.5-1.5B}" \
    --big_device "${BIG_DEVICE:-cuda:0}" \
    --small_device "${SMALL_DEVICE:-cuda:0}" \
    --attn_implementation "${ATTN_IMPLEMENTATION:-sdpa}" \
    --dataset_name "${DATASET_NAME:-wikitext}" \
    --dataset_config "${DATASET_CONFIG:-wikitext-2-raw-v1}" \
    --eval_split "${EVAL_SPLIT:-validation}" \
    --prompt_len "$context" \
    --num_prompts "$num_eval" \
    --warmup_prompts "$warmup_prompts" \
    --skip_prompts "$skip_blocks" \
    --draft_steps "${DRAFT_STEPS:-4}" \
    --max_new_tokens "${MAX_NEW_TOKENS:-16}" \
    --seed "$seed" \
    "${shuffle_args[@]}" \
    --quant_configs "$quant_configs" \
    --out_dir "$out_dir" \
    --wandb_run_name "acceptance_b${budget}_c${context}_s${seed}" \
    "${stream_args[@]}" \
    "${text_column_args[@]}" \
    "${common_wandb[@]}"
else
  echo "Unknown objective: $objective" >&2
  exit 1
fi
