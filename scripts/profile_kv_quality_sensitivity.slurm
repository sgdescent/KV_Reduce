#!/bin/bash
#SBATCH --job-name=kv_quality_sens
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
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
MODEL="${MODEL:-Qwen/Qwen2.5-1.5B}"
DATASET_NAME="${DATASET_NAME:-wikitext}"
DATASET_CONFIG="${DATASET_CONFIG:-wikitext-2-raw-v1}"
EVAL_SPLIT="${EVAL_SPLIT:-validation}"
PROMPT_LEN="${PROMPT_LEN:-1024}"
CONTINUATION_LEN="${CONTINUATION_LEN:-32}"
NUM_SEQUENCES="${NUM_SEQUENCES:-32}"
SEED="${SEED:-0}"
LAYERS="${LAYERS:-top:8}"
COMPONENTS="${COMPONENTS:-k,v}"
BITS="${BITS:-8,4}"
QUANT_CONFIGS="${QUANT_CONFIGS:-}"
QUALITY_RISK_METRIC="${QUALITY_RISK_METRIC:-kl}"
OUT_DIR="${OUT_DIR:-outputs/kv_quality_sensitivity_${SLURM_JOB_ID:-local}}"
WANDB_PROJECT="${WANDB_PROJECT:-kv-reduce}"
WANDB_GROUP="${WANDB_GROUP:-kv-objective-quality}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-kv_quality_${SLURM_JOB_ID:-local}}"

cmd=(
  "$PYTHON_BIN" -u profile_kv_quality_sensitivity.py
  --model "$MODEL"
  --device cuda:0
  --dataset_name "$DATASET_NAME"
  --dataset_config "$DATASET_CONFIG"
  --eval_split "$EVAL_SPLIT"
  --prompt_len "$PROMPT_LEN"
  --continuation_len "$CONTINUATION_LEN"
  --num_sequences "$NUM_SEQUENCES"
  --seed "$SEED"
  --quality_risk_metric "$QUALITY_RISK_METRIC"
  --out_dir "$OUT_DIR"
)

if [[ -n "$QUANT_CONFIGS" ]]; then
  cmd+=(--quant_configs "$QUANT_CONFIGS")
else
  cmd+=(--layers "$LAYERS" --components "$COMPONENTS" --bits "$BITS")
fi
if [[ "${STREAM_EVAL:-0}" == "1" ]]; then
  cmd+=(--stream_eval)
fi
if [[ "${SHUFFLE_EVAL:-0}" == "1" ]]; then
  cmd+=(--shuffle_eval)
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
