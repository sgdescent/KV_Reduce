#!/bin/bash
#SBATCH --job-name=kv_verify_diag
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
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
MODEL="${MODEL:-Qwen/Qwen2.5-3B}"
SMALL_MODEL="${SMALL_MODEL:-Qwen/Qwen2.5-1.5B}"
DATASET_NAME="${DATASET_NAME:-wikitext}"
DATASET_CONFIG="${DATASET_CONFIG:-wikitext-2-raw-v1}"
EVAL_SPLIT="${EVAL_SPLIT:-validation}"
PROMPT_LEN="${PROMPT_LEN:-1024}"
NUM_PROMPTS="${NUM_PROMPTS:-1}"
SKIP_PROMPTS="${SKIP_PROMPTS:-0}"
DRAFT_STEPS="${DRAFT_STEPS:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
SEED="${SEED:-0}"
OUT="${OUT:-outputs/target_verification_diagnostic/seed_${SEED}_skip_${SKIP_PROMPTS}.json}"

cmd=(
  "$PYTHON_BIN" -u diagnose_cached_target_verification.py
  --model "$MODEL"
  --small_model "$SMALL_MODEL"
  --device cuda:0
  --dtype bf16
  --attn_implementation sdpa
  --dataset_name "$DATASET_NAME"
  --dataset_config "$DATASET_CONFIG"
  --eval_split "$EVAL_SPLIT"
  --prompt_len "$PROMPT_LEN"
  --num_prompts "$NUM_PROMPTS"
  --skip_prompts "$SKIP_PROMPTS"
  --draft_steps "$DRAFT_STEPS"
  --max_new_tokens "$MAX_NEW_TOKENS"
  --seed "$SEED"
  --run_speculative_audit
  --out "$OUT"
)

if [[ "${SHUFFLE_EVAL:-1}" == "1" ]]; then
  cmd+=(--shuffle_eval)
fi

printf 'Running command:\n'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
