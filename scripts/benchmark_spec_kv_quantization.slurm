#!/bin/bash
#SBATCH --job-name=kv_spec_quant
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs outputs

VENV_PATH="${VENV_PATH:-}"
if [[ -z "$VENV_PATH" ]]; then
  for candidate in .venv venv env; do
    if [[ -f "$candidate/bin/activate" ]]; then
      VENV_PATH="$candidate"
      break
    fi
  done
fi
if [[ -n "$VENV_PATH" ]]; then
  # shellcheck disable=SC1090
  source "$VENV_PATH/bin/activate"
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
BIG_MODEL="${BIG_MODEL:-Qwen/Qwen2.5-3B}"
SMALL_MODEL="${SMALL_MODEL:-Qwen/Qwen2.5-1.5B}"
DATASET_NAME="${DATASET_NAME:-wikitext}"
DATASET_CONFIG="${DATASET_CONFIG:-wikitext-2-raw-v1}"
EVAL_SPLIT="${EVAL_SPLIT:-validation}"
PROMPT_LEN="${PROMPT_LEN:-1024}"
NUM_PROMPTS="${NUM_PROMPTS:-100}"
WARMUP_PROMPTS="${WARMUP_PROMPTS:-5}"
DRAFT_STEPS="${DRAFT_STEPS:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
SEED="${SEED:-0}"
QUANT_CONFIGS="${QUANT_CONFIGS:-none,8,k8v4,k4v8,k4v4}"
TARGET_QUANT_CONFIGS="${TARGET_QUANT_CONFIGS:-none}"
KEY_QUANT_AXIS="${KEY_QUANT_AXIS:-per_token}"
KEY_GROUP_SIZE="${KEY_GROUP_SIZE:-32}"
KEY_RESIDUAL_LENGTH="${KEY_RESIDUAL_LENGTH:-128}"
VALUE_QUANT_SCHEME="${VALUE_QUANT_SCHEME:-symmetric}"
OUT_DIR="${OUT_DIR:-outputs/spec_kv_quant_benchmark_${SLURM_JOB_ID:-local}}"
WANDB_PROJECT="${WANDB_PROJECT:-kv-reduce}"
WANDB_GROUP="${WANDB_GROUP:-spec-kv-quant}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-spec_kv_quant_${SLURM_JOB_ID:-local}}"

cmd=(
  "$PYTHON_BIN" -u benchmark_spec_kv_quantization.py
  --big_model "$BIG_MODEL"
  --small_model "$SMALL_MODEL"
  --big_device cuda:0
  --small_device cuda:0
  --dataset_name "$DATASET_NAME"
  --dataset_config "$DATASET_CONFIG"
  --eval_split "$EVAL_SPLIT"
  --prompt_len "$PROMPT_LEN"
  --num_prompts "$NUM_PROMPTS"
  --warmup_prompts "$WARMUP_PROMPTS"
  --draft_steps "$DRAFT_STEPS"
  --max_new_tokens "$MAX_NEW_TOKENS"
  --seed "$SEED"
  --quant_configs "$QUANT_CONFIGS"
  --target_quant_configs "$TARGET_QUANT_CONFIGS"
  --key_quant_axis "$KEY_QUANT_AXIS"
  --key_group_size "$KEY_GROUP_SIZE"
  --key_residual_length "$KEY_RESIDUAL_LENGTH"
  --value_quant_scheme "$VALUE_QUANT_SCHEME"
  --out_dir "$OUT_DIR"
)

if [[ "${STREAM_EVAL:-0}" == "1" ]]; then
  cmd+=(--stream_eval)
fi
if [[ "${SHUFFLE_EVAL:-0}" == "1" ]]; then
  cmd+=(--shuffle_eval)
fi
if [[ "${ALLOW_INCOMPATIBLE_TOKENIZERS:-0}" == "1" ]]; then
  cmd+=(--allow_incompatible_tokenizers)
fi
if [[ "${ENABLE_WANDB:-1}" == "1" ]]; then
  cmd+=(--wandb --wandb_project "$WANDB_PROJECT" --wandb_group "$WANDB_GROUP" --wandb_run_name "$WANDB_RUN_NAME")
  if [[ -n "${WANDB_ENTITY:-}" ]]; then
    cmd+=(--wandb_entity "$WANDB_ENTITY")
  fi
fi

printf 'Running command:\n'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
