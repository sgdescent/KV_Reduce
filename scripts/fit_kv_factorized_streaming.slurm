#!/bin/bash
#SBATCH --job-name=kv_factorized
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --partition=general
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Runs fit_kv_factorized_probe.py on one node with two GPUs:
#   - big model forward pass on cuda:0
#   - small model forward pass + translator training on cuda:1
#
# Example:
#   sbatch scripts/fit_kv_factorized_streaming.slurm
#
# Override defaults at submit time if you want a larger streamed corpus:
#   sbatch --export=ALL,DATASET_NAME=HuggingFaceFW/fineweb-edu,DATASET_CONFIG=none,TRAIN_SEQUENCES=50000 \
#     scripts/fit_kv_factorized_streaming.slurm
#
# W&B:
#   Export WANDB_API_KEY before submission, or run `wandb login` once in the same env on a login node.
#   Disable W&B for a job with:
#     sbatch --export=ALL,ENABLE_WANDB=0 scripts/fit_kv_factorized_streaming.slurm
#
# Python env:
#   If you have a venv, submit with:
#     sbatch --export=ALL,VENV_PATH=/path/to/venv scripts/fit_kv_factorized_streaming.slurm
#   The script also auto-detects ./.venv, ./venv, and ./env if present.

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
  if [[ ! -f "$VENV_PATH/bin/activate" ]]; then
    echo "VENV_PATH=$VENV_PATH does not contain bin/activate" >&2
    exit 1
  fi
  # shellcheck disable=SC1090
  source "$VENV_PATH/bin/activate"
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! "$PYTHON_BIN" -c "import torch" >/dev/null 2>&1; then
  echo "PyTorch is not available in $PYTHON_BIN." >&2
  echo "Use VENV_PATH=/path/to/venv (or create .venv in the repo) so the job can import torch." >&2
  exit 1
fi

BIG_MODEL="${BIG_MODEL:-Qwen/Qwen2.5-3B}"
SMALL_MODEL="${SMALL_MODEL:-Qwen/Qwen2.5-1.5B}"
BIG_DTYPE="${BIG_DTYPE:-bf16}"
SMALL_DTYPE="${SMALL_DTYPE:-bf16}"

DATASET_NAME="${DATASET_NAME:-wikitext}"
DATASET_CONFIG="${DATASET_CONFIG:-wikitext-2-raw-v1}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
EVAL_SPLIT="${EVAL_SPLIT:-validation}"
EVAL_DATASET_NAME="${EVAL_DATASET_NAME:-}"
EVAL_DATASET_CONFIG="${EVAL_DATASET_CONFIG:-}"
EVAL_SPLIT_FALLBACKS="${EVAL_SPLIT_FALLBACKS:-test,train}"
TRAIN_SKIP_BLOCKS="${TRAIN_SKIP_BLOCKS:-0}"
EVAL_SKIP_BLOCKS="${EVAL_SKIP_BLOCKS:-0}"
SAME_DATASET_HOLDOUT_BLOCKS="${SAME_DATASET_HOLDOUT_BLOCKS:-0}"

SEQ_LEN="${SEQ_LEN:-256}"
TRAIN_SEQUENCES="${TRAIN_SEQUENCES:-20000}"
EVAL_SEQUENCES="${EVAL_SEQUENCES:-128}"
POSITION_STRIDE="${POSITION_STRIDE:-2}"
MAX_ROWS_PER_LAYER_PER_BLOCK="${MAX_ROWS_PER_LAYER_PER_BLOCK:-128}"
RANK="${RANK:-64}"
K_RANK="${K_RANK:-$RANK}"
V_RANK="${V_RANK:-$RANK}"
HIDDEN_LAYERS="${HIDDEN_LAYERS:-0}"
K_HIDDEN_LAYERS="${K_HIDDEN_LAYERS:-$HIDDEN_LAYERS}"
V_HIDDEN_LAYERS="${V_HIDDEN_LAYERS:-$HIDDEN_LAYERS}"
LR="${LR:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
COS_LOSS_WEIGHT="${COS_LOSS_WEIGHT:-0.1}"
GRAD_ACCUM_SEQUENCES="${GRAD_ACCUM_SEQUENCES:-1}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
WARMUP_STEPS="${WARMUP_STEPS:-200}"
MIN_LR_RATIO="${MIN_LR_RATIO:-0.1}"
SHUFFLE_BUFFER_SIZE="${SHUFFLE_BUFFER_SIZE:-10000}"
TRAIN_TARGETS="${TRAIN_TARGETS:-both}"
CHECKPOINT_EVERY_SEQUENCES="${CHECKPOINT_EVERY_SEQUENCES:-1000}"
EVAL_EVERY_SEQUENCES="${EVAL_EVERY_SEQUENCES:-0}"
PERIODIC_EVAL_SEQUENCES="${PERIODIC_EVAL_SEQUENCES:-0}"
AUTO_RESUME="${AUTO_RESUME:-1}"
RESUME_FROM="${RESUME_FROM:-}"
RESUME_OPTIMIZER="${RESUME_OPTIMIZER:-1}"
TOPK="${TOPK:-5}"
SEED="${SEED:-0}"

OUT_DIR="${OUT_DIR:-outputs/qwen25_3b_to_15b_factorized_${SLURM_JOB_ID:-local}}"
WANDB_PROJECT="${WANDB_PROJECT:-kv-reduce}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-qwen25_3b_15b_factorized_${SLURM_JOB_ID:-local}}"

cmd=(
  "$PYTHON_BIN"
  fit_kv_factorized_probe.py
  --big_model "$BIG_MODEL"
  --small_model "$SMALL_MODEL"
  --big_device cuda:0
  --small_device cuda:1
  --big_dtype "$BIG_DTYPE"
  --small_dtype "$SMALL_DTYPE"
  --dataset_name "$DATASET_NAME"
  --dataset_config "$DATASET_CONFIG"
  --train_split "$TRAIN_SPLIT"
  --eval_split "$EVAL_SPLIT"
  --eval_split_fallbacks "$EVAL_SPLIT_FALLBACKS"
  --train_skip_blocks "$TRAIN_SKIP_BLOCKS"
  --eval_skip_blocks "$EVAL_SKIP_BLOCKS"
  --same_dataset_holdout_blocks "$SAME_DATASET_HOLDOUT_BLOCKS"
  --stream_train
  --seq_len "$SEQ_LEN"
  --train_sequences "$TRAIN_SEQUENCES"
  --eval_sequences "$EVAL_SEQUENCES"
  --position_stride "$POSITION_STRIDE"
  --max_rows_per_layer_per_block "$MAX_ROWS_PER_LAYER_PER_BLOCK"
  --rank "$RANK"
  --k_rank "$K_RANK"
  --v_rank "$V_RANK"
  --hidden_layers "$HIDDEN_LAYERS"
  --k_hidden_layers "$K_HIDDEN_LAYERS"
  --v_hidden_layers "$V_HIDDEN_LAYERS"
  --lr "$LR"
  --weight_decay "$WEIGHT_DECAY"
  --cos_loss_weight "$COS_LOSS_WEIGHT"
  --grad_accum_sequences "$GRAD_ACCUM_SEQUENCES"
  --grad_clip "$GRAD_CLIP"
  --warmup_steps "$WARMUP_STEPS"
  --min_lr_ratio "$MIN_LR_RATIO"
  --checkpoint_every_sequences "$CHECKPOINT_EVERY_SEQUENCES"
  --eval_every_sequences "$EVAL_EVERY_SEQUENCES"
  --periodic_eval_sequences "$PERIODIC_EVAL_SEQUENCES"
  --shuffle_train
  --shuffle_buffer_size "$SHUFFLE_BUFFER_SIZE"
  --train_targets "$TRAIN_TARGETS"
  --topk "$TOPK"
  --seed "$SEED"
  --out_dir "$OUT_DIR"
)

if [[ -n "$EVAL_DATASET_NAME" ]]; then
  cmd+=(--eval_dataset_name "$EVAL_DATASET_NAME")
fi

if [[ -n "$EVAL_DATASET_CONFIG" ]]; then
  cmd+=(--eval_dataset_config "$EVAL_DATASET_CONFIG")
fi

if [[ "$AUTO_RESUME" == "1" ]]; then
  cmd+=(--auto_resume)
fi

if [[ -n "$RESUME_FROM" ]]; then
  cmd+=(--resume_from "$RESUME_FROM")
fi

if [[ "$RESUME_OPTIMIZER" == "0" ]]; then
  cmd+=(--no_resume_optimizer)
fi

if [[ "${STREAM_EVAL:-0}" == "1" ]]; then
  cmd+=(--stream_eval)
fi

if [[ "${USE_LAYERNORM:-0}" == "1" ]]; then
  cmd+=(--use_layernorm)
fi

if [[ "${USE_BIAS:-0}" == "1" ]]; then
  cmd+=(--use_bias)
fi

if [[ "${NO_RESIDUAL:-0}" == "1" ]]; then
  cmd+=(--no_residual)
fi

if [[ "${ALLOW_INCOMPATIBLE_TOKENIZERS:-0}" == "1" ]]; then
  cmd+=(--allow_incompatible_tokenizers)
fi

if [[ "${ENABLE_WANDB:-1}" == "1" ]]; then
  cmd+=(
    --wandb
    --wandb_project "$WANDB_PROJECT"
    --wandb_run_name "$WANDB_RUN_NAME"
  )
  if [[ -n "${WANDB_ENTITY:-}" ]]; then
    cmd+=(--wandb_entity "$WANDB_ENTITY")
  fi
  if [[ -n "${WANDB_GROUP:-}" ]]; then
    cmd+=(--wandb_group "$WANDB_GROUP")
  fi
fi

printf 'Running command:\n'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
