#!/bin/bash
#SBATCH --job-name=kv_verify_matrix
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
cd "${REPO_ROOT:-/home/sakshamg/KV_Reduce}"
mkdir -p logs outputs
if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

backends=(sdpa eager sdpa eager)
reset_shadow=(0 0 1 1)
labels=(sdpa eager sdpa_shadow_reset eager_shadow_reset)
task_id="${SLURM_ARRAY_TASK_ID:-0}"
backend="${backends[$task_id]}"
reset="${reset_shadow[$task_id]}"
label="${labels[$task_id]}"
root="${OUT_ROOT:-outputs/target_verification_diagnostic/model_matrix}"

cmd=(
  python3 -u diagnose_cached_target_verification.py
  --model "${MODEL:-Qwen/Qwen2.5-3B}"
  --small_model "${SMALL_MODEL:-Qwen/Qwen2.5-1.5B}"
  --device cuda:0
  --dtype "${DTYPE:-bf16}"
  --attn_implementation "$backend"
  --dataset_name "${DATASET_NAME:-wikitext}"
  --dataset_config "${DATASET_CONFIG:-wikitext-2-raw-v1}"
  --eval_split "${EVAL_SPLIT:-validation}"
  --eval_split_fallbacks "${EVAL_SPLIT_FALLBACKS:-test,train}"
  --prompt_len "${PROMPT_LEN:-1024}"
  --num_prompts "${NUM_PROMPTS:-64}"
  --skip_prompts "${SKIP_PROMPTS:-0}"
  --draft_steps "${DRAFT_STEPS:-4}"
  --max_new_tokens "${MAX_NEW_TOKENS:-16}"
  --seed "${SEED:-0}"
  --skip_unit_audit
  --run_speculative_audit
  --out "$root/raw/${label}.json"
)

if [[ "${STREAM_EVAL:-0}" == "1" ]]; then
  cmd+=(--stream_eval)
fi
if [[ "${SHUFFLE_EVAL:-0}" == "1" ]]; then
  cmd+=(--shuffle_eval)
fi
if [[ "$reset" == "1" ]]; then
  cmd+=(--reset_target_from_sequential_shadow)
fi

export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

printf 'Running command:'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"

