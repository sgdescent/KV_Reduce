#!/bin/bash
#SBATCH --job-name=kv_abs_bench
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --partition=general
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
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
TRANSLATOR_PATH="${TRANSLATOR_PATH:-outputs/kv_absorbed/absorbed_translator.pt}"
DATASET_NAME="${DATASET_NAME:-wikitext}"
DATASET_CONFIG="${DATASET_CONFIG:-wikitext-2-raw-v1}"
PROMPT_LEN="${PROMPT_LEN:-1024}"
NUM_PROMPTS="${NUM_PROMPTS:-200}"
WARMUP_PROMPTS="${WARMUP_PROMPTS:-10}"
DRAFT_STEPS="${DRAFT_STEPS:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
SHARED_LAYERS="${SHARED_LAYERS:-top:4}"
SHARED_VARIANT="${SHARED_VARIANT:-full}"
NORM_MATCH="${NORM_MATCH:-rms}"
ABSORBED_CACHE_MODE="${ABSORBED_CACHE_MODE:-prefix}"
OUT_DIR="${OUT_DIR:-outputs/benchmark_absorbed_cached_${SLURM_JOB_ID:-local}}"
WANDB_PROJECT="${WANDB_PROJECT:-kv-reduce}"
WANDB_GROUP="${WANDB_GROUP:-cached-prefix-benchmark}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-cached_prefix_benchmark_${SLURM_JOB_ID:-local}}"

cmd=(
  "$PYTHON_BIN" benchmark_absorbed_spec_decode.py
  --big_model "$BIG_MODEL"
  --small_model "$SMALL_MODEL"
  --translator_path "$TRANSLATOR_PATH"
  --big_device cuda:0
  --small_device cuda:1
  --dataset_name "$DATASET_NAME"
  --dataset_config "$DATASET_CONFIG"
  --prompt_len "$PROMPT_LEN"
  --num_prompts "$NUM_PROMPTS"
  --warmup_prompts "$WARMUP_PROMPTS"
  --draft_steps "$DRAFT_STEPS"
  --max_new_tokens "$MAX_NEW_TOKENS"
  --shared_layers "$SHARED_LAYERS"
  --shared_variant "$SHARED_VARIANT"
  --norm_match "$NORM_MATCH"
  --absorbed_cache_mode "$ABSORBED_CACHE_MODE"
  --modes native,absorbed
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
