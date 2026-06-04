#!/bin/bash
#SBATCH --job-name=kv_layer_map
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
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
DATASET_NAME="${DATASET_NAME:-HuggingFaceFW/fineweb-edu}"
DATASET_CONFIG="${DATASET_CONFIG:-sample-10BT}"
SEQ_LEN="${SEQ_LEN:-512}"
NUM_SEQUENCES="${NUM_SEQUENCES:-128}"
POSITION_STRIDE="${POSITION_STRIDE:-8}"
MAX_ROWS_PER_LAYER="${MAX_ROWS_PER_LAYER:-512}"
OUT_DIR="${OUT_DIR:-outputs/layer_map_${SLURM_JOB_ID:-local}}"
WANDB_PROJECT="${WANDB_PROJECT:-kv-reduce}"
WANDB_GROUP="${WANDB_GROUP:-layer-map}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-layer_map_${SLURM_JOB_ID:-local}}"

cmd=(
  "$PYTHON_BIN" learn_layer_map.py
  --big_model "$BIG_MODEL"
  --small_model "$SMALL_MODEL"
  --big_device cuda:0
  --small_device cuda:1
  --dataset_name "$DATASET_NAME"
  --dataset_config "$DATASET_CONFIG"
  --seq_len "$SEQ_LEN"
  --num_sequences "$NUM_SEQUENCES"
  --position_stride "$POSITION_STRIDE"
  --max_rows_per_layer "$MAX_ROWS_PER_LAYER"
  --streaming
  --out_dir "$OUT_DIR"
)

if [[ "${SHUFFLE:-1}" == "1" ]]; then
  cmd+=(--shuffle)
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
