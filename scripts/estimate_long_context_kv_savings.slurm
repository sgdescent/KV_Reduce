#!/bin/bash
#SBATCH --job-name=kv_savings
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
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
BIG_MODEL="${BIG_MODEL:-Qwen/Qwen2.5-7B}"
SMALL_MODEL="${SMALL_MODEL:-Qwen/Qwen2.5-3B}"
CONTEXTS="${CONTEXTS:-512,1024,4096,8192,16384,32768}"
SHARED_LAYERS_SPECS="${SHARED_LAYERS_SPECS:-top:4,top:8,all}"
DRAFT_TAIL_LEN="${DRAFT_TAIL_LEN:-4}"
OUT_DIR="${OUT_DIR:-outputs/long_context_savings_${SLURM_JOB_ID:-local}}"

cmd=(
  "$PYTHON_BIN" estimate_long_context_kv_savings.py
  --big_model "$BIG_MODEL"
  --small_model "$SMALL_MODEL"
  --contexts "$CONTEXTS"
  --shared_layers_specs "$SHARED_LAYERS_SPECS"
  --draft_tail_len "$DRAFT_TAIL_LEN"
  --out_dir "$OUT_DIR"
)

printf 'Running command:\n'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
