#!/bin/bash
#SBATCH --job-name=spec_kv_xfam
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

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs outputs/iclr_spec_kv

VENV_PATH="${VENV_PATH:-.venv}"
if [[ -f "$VENV_PATH/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "$VENV_PATH/bin/activate"
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-${TASK_ID:-0}}"
STAGE="${STAGE:-smoke}"
ENABLE_WANDB="${ENABLE_WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-kv-reduce}"
DATASET_NAME="${DATASET_NAME:-wikitext}"
DATASET_CONFIG="${DATASET_CONFIG:-wikitext-2-raw-v1}"
EVAL_SPLIT="${EVAL_SPLIT:-validation}"
SEED="${SEED:-0}"
QUANT_CONFIGS="${QUANT_CONFIGS:-none,8,k8v4,k4v8,k4v4}"
BIG_DEVICE="${BIG_DEVICE:-cuda:0}"
SMALL_DEVICE="${SMALL_DEVICE:-cuda:0}"

export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

case "$TASK_ID" in
  0)
    PAIR_NAME="qwen25_3b_15b"
    BIG_MODEL="Qwen/Qwen2.5-3B"
    SMALL_MODEL="Qwen/Qwen2.5-1.5B"
    ;;
  1)
    PAIR_NAME="qwen25_7b_3b"
    BIG_MODEL="Qwen/Qwen2.5-7B"
    SMALL_MODEL="Qwen/Qwen2.5-3B"
    ;;
  2)
    PAIR_NAME="qwen3_8b_4b"
    BIG_MODEL="Qwen/Qwen3-8B"
    SMALL_MODEL="Qwen/Qwen3-4B"
    ;;
  3)
    PAIR_NAME="llama31_8b_llama32_3b"
    BIG_MODEL="meta-llama/Llama-3.1-8B"
    SMALL_MODEL="meta-llama/Llama-3.2-3B"
    ;;
  4)
    PAIR_NAME="olmo2_7b_1b"
    BIG_MODEL="allenai/OLMo-2-1124-7B"
    SMALL_MODEL="allenai/OLMo-2-0425-1B"
    ;;
  5)
    PAIR_NAME="smollm2_17b_360m"
    BIG_MODEL="HuggingFaceTB/SmolLM2-1.7B"
    SMALL_MODEL="HuggingFaceTB/SmolLM2-360M"
    ;;
  *)
    echo "Unsupported task id $TASK_ID; expected 0-5." >&2
    exit 2
    ;;
esac

BASE_OUT="outputs/iclr_spec_kv/${PAIR_NAME}"
mkdir -p "$BASE_OUT"

WANDB_ARGS=()
if [[ "$ENABLE_WANDB" == "1" ]]; then
  WANDB_ARGS=(--wandb --wandb_project "$WANDB_PROJECT")
fi

common_args=(
  --big_model "$BIG_MODEL"
  --small_model "$SMALL_MODEL"
  --big_device "$BIG_DEVICE"
  --small_device "$SMALL_DEVICE"
  --big_dtype bf16
  --small_dtype bf16
  --dataset_name "$DATASET_NAME"
  --dataset_config "$DATASET_CONFIG"
  --eval_split "$EVAL_SPLIT"
  --draft_steps 4
  --max_new_tokens 16
  --seed "$SEED"
)

run_benchmark() {
  local context_len="$1"
  local num_prompts="$2"
  local warmup_prompts="$3"
  local label="$4"
  local configs="$5"
  shift 5
  local out_dir="$BASE_OUT/$label"

  "$PYTHON_BIN" -u benchmark_spec_kv_quantization.py \
    "${common_args[@]}" \
    --prompt_len "$context_len" \
    --num_prompts "$num_prompts" \
    --warmup_prompts "$warmup_prompts" \
    --quant_configs "$configs" \
    --out_dir "$out_dir" \
    "${WANDB_ARGS[@]}" \
    --wandb_group "iclr-spec-kv-${STAGE}" \
    --wandb_run_name "${PAIR_NAME}_${label}" \
    "$@"
}

echo "stage=$STAGE pair=$PAIR_NAME big=$BIG_MODEL@$BIG_DEVICE small=$SMALL_MODEL@$SMALL_DEVICE seed=$SEED"

require_artifact() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "Skipping $STAGE for $PAIR_NAME because prerequisite is missing: $path"
    exit 0
  fi
}

case "$STAGE" in
  smoke)
    # Small enough to validate model access, cache APIs, and tokenizer compatibility.
    common_args+=(--max_new_tokens 8)
    run_benchmark 256 2 1 "smoke" "none,k8v4,k4v8"
    ;;

  main)
    require_artifact "$BASE_OUT/smoke/summary.json"
    run_benchmark 1024 "${PROMPTS_1K:-32}" 3 "wikitext_ctx1024" "$QUANT_CONFIGS"
    run_benchmark 4096 "${PROMPTS_4K:-12}" 2 "wikitext_ctx4096" "$QUANT_CONFIGS"
    ;;

  sensitivity)
    require_artifact "$BASE_OUT/wikitext_ctx1024/summary.json"
    require_artifact "$BASE_OUT/wikitext_ctx4096/summary.json"
    PROFILE_OUT="$BASE_OUT/sensitivity_top8_ctx1024"
    "$PYTHON_BIN" -u profile_spec_kv_sensitivity.py \
      "${common_args[@]}" \
      --prompt_len 1024 \
      --num_prompts "${SENSITIVITY_PROMPTS:-8}" \
      --warmup_prompts 2 \
      --layers top:8 \
      --components k,v \
      --bits 8,4 \
      --out_dir "$PROFILE_OUT" \
      "${WANDB_ARGS[@]}" \
      --wandb_group "iclr-spec-kv-sensitivity" \
      --wandb_run_name "${PAIR_NAME}_sensitivity_top8_ctx1024_seed${SEED}"
    ;;

  allocation)
    PROFILE_OUT="$BASE_OUT/sensitivity_top8_ctx1024"
    ALLOCATION_OUT="$BASE_OUT/allocation_top8_ctx1024"
    PROFILE_CSV="$PROFILE_OUT/profile_summary.csv"
    if [[ ! -f "$PROFILE_CSV" ]]; then
      echo "Missing sensitivity profile: $PROFILE_CSV" >&2
      exit 3
    fi
    "$PYTHON_BIN" -u search_kv_bit_allocation.py \
      --profile_csv "$PROFILE_CSV" \
      --allowed_bits 4,8,16 \
      --max_component_drop "${MAX_COMPONENT_DROP:-0.01}" \
      --max_total_drop "${MAX_TOTAL_DROP:-0.05}" \
      --name sensitivity_aware \
      --out_dir "$ALLOCATION_OUT"
    run_benchmark 1024 "${ALLOCATION_PROMPTS:-32}" 3 "allocation_ctx1024" \
      "none,allocation:${ALLOCATION_OUT}/allocation.json"
    ;;

  robustness)
    require_artifact "$BASE_OUT/wikitext_ctx1024/summary.json"
    # Held-out distribution and prompt-order robustness. Duplicate argparse
    # options below intentionally override the WikiText defaults in common_args.
    for robust_seed in 1 2; do
      run_benchmark 1024 "${ROBUSTNESS_PROMPTS:-16}" 2 "c4_ctx1024_seed${robust_seed}" \
        "none,k8v4,k4v8" \
        --dataset_name allenai/c4 \
        --dataset_config en \
        --eval_split validation \
        --stream_eval \
        --shuffle_eval \
        --seed "$robust_seed"
    done
    ;;

  long_context)
    require_artifact "$BASE_OUT/wikitext_ctx4096/summary.json"
    # Exploratory sample counts keep full-prefix target verification tractable.
    run_benchmark 8192 "${PROMPTS_8K:-4}" 1 "wikitext_ctx8192" "none,k8v4,k4v8"
    run_benchmark 16384 "${PROMPTS_16K:-2}" 1 "wikitext_ctx16384" "none,k8v4,k4v8"
    ;;

  *)
    echo "Unsupported stage $STAGE; use smoke, main, sensitivity, allocation, robustness, or long_context." >&2
    exit 2
    ;;
esac
