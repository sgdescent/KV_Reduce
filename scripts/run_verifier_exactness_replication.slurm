#!/bin/bash
#SBATCH --job-name=kv_verify_power
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
cd "${REPO_ROOT:-/home/sakshamg/KV_Reduce}"
mkdir -p logs outputs
if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

dtypes=(bf16 bf16 float32 float32)
backends=(sdpa eager sdpa eager)
task_id="${SLURM_ARRAY_TASK_ID:-0}"
dtype="${dtypes[$task_id]}"
backend="${backends[$task_id]}"
root="${OUT_ROOT:-outputs/target_verification_diagnostic/powered_backend_dtype}"
out="$root/${dtype}_${backend}.json"

export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

python3 -u diagnose_cached_target_verification.py \
  --model Qwen/Qwen2.5-3B \
  --small_model Qwen/Qwen2.5-1.5B \
  --device cuda:0 \
  --dtype "$dtype" \
  --attn_implementation "$backend" \
  --dataset_name wikitext \
  --dataset_config wikitext-2-raw-v1 \
  --eval_split validation \
  --shuffle_eval \
  --prompt_len 1024 \
  --num_prompts "${NUM_PROMPTS:-32}" \
  --draft_steps 4 \
  --max_new_tokens 16 \
  --skip_unit_audit \
  --run_speculative_audit \
  --seed 73 \
  --out "$out"
