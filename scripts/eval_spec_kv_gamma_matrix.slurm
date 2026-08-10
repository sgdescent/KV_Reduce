#!/bin/bash
#SBATCH --job-name=kv_gamma_eval
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
IFS=$'\t' read -r budget context seed draft_steps num_eval quant_configs out_dir <<< "$line"
mkdir -p "$out_dir"

wandb_args=()
if [[ "${ENABLE_WANDB:-0}" == "1" ]]; then
  wandb_args=(
    --wandb
    --wandb_project "${WANDB_PROJECT:-kv-reduce}"
    --wandb_group "${WANDB_GROUP:-objective-gamma}"
    --wandb_run_name "gamma_b${budget}_c${context}_g${draft_steps}_s${seed}"
  )
fi

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
  --draft_steps "$draft_steps" \
  --max_new_tokens "${MAX_NEW_TOKENS:-16}" \
  --seed "$seed" \
  --shuffle_eval \
  --quant_configs "$quant_configs" \
  --out_dir "$out_dir" \
  "${wandb_args[@]}"
