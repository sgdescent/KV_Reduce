#!/bin/bash
#SBATCH --job-name=kv_abs_fit
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=24:00:00
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
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
BIG_MODEL="${BIG_MODEL:-Qwen/Qwen2.5-3B}"
SMALL_MODEL="${SMALL_MODEL:-Qwen/Qwen2.5-1.5B}"
DATASET_NAME="${DATASET_NAME:-HuggingFaceFW/fineweb-edu}"
DATASET_CONFIG="${DATASET_CONFIG:-sample-10BT}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
SEQ_LEN="${SEQ_LEN:-512}"
TRAIN_SEQUENCES="${TRAIN_SEQUENCES:-50000}"
K_TRAIN_SEQUENCES="${K_TRAIN_SEQUENCES:-}"
O_TRAIN_SEQUENCES="${O_TRAIN_SEQUENCES:-}"
LAMBDA_REG="${LAMBDA_REG:-1e-4}"
OUTPUT_ROUTING_SOURCE="${OUTPUT_ROUTING_SOURCE:-shared}"
LAYER_MAP_FILE="${LAYER_MAP_FILE:-}"
OUT_DIR="${OUT_DIR:-outputs/kv_absorbed_learned_${SLURM_JOB_ID:-local}}"
WANDB_PROJECT="${WANDB_PROJECT:-kv-reduce}"
WANDB_GROUP="${WANDB_GROUP:-absorbed-fit}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-absorbed_fit_${SLURM_JOB_ID:-local}}"

cmd=(
  "$PYTHON_BIN" -u fit_kv_absorbed.py
  --big_model "$BIG_MODEL"
  --small_model "$SMALL_MODEL"
  --big_device cuda:0
  --small_device cuda:1
  --dataset_name "$DATASET_NAME"
  --dataset_config "$DATASET_CONFIG"
  --train_split "$TRAIN_SPLIT"
  --seq_len "$SEQ_LEN"
  --train_sequences "$TRAIN_SEQUENCES"
  --lambda_reg "$LAMBDA_REG"
  --output_routing_source "$OUTPUT_ROUTING_SOURCE"
  --stream_train
  --out_dir "$OUT_DIR"
)

if [[ -n "$K_TRAIN_SEQUENCES" ]]; then
  cmd+=(--k_train_sequences "$K_TRAIN_SEQUENCES")
fi

if [[ -n "$O_TRAIN_SEQUENCES" ]]; then
  cmd+=(--o_train_sequences "$O_TRAIN_SEQUENCES")
fi

if [[ -n "$LAYER_MAP_FILE" ]]; then
  cmd+=(--layer_map_file "$LAYER_MAP_FILE")
fi
if [[ "${SHUFFLE_TRAIN:-1}" == "1" ]]; then
  cmd+=(--shuffle_train)
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
