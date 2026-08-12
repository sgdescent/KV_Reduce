#!/bin/bash
#SBATCH --job-name=kv_passkey_lagg
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --exclude=catalyst-0-[9,15]
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
cd "${REPO_ROOT:-/home/sakshamg/KV_Reduce}"
mkdir -p logs
if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

task_id=$((10#${SLURM_ARRAY_TASK_ID:-0}))
case "$task_id" in
  0) model_tag="qwen25_15b" ;;
  1) model_tag="llama32_3b" ;;
  2) model_tag="qwen3_4b" ;;
  *) echo "Unsupported model index: $task_id" >&2; exit 2 ;;
esac

context="${PASSKEY_CONTEXT:-16384}"
root="${OUT_ROOT:-outputs/passkey_layer_sensitivity_v1}/$model_tag"
python3 -u aggregate_passkey_layer_sensitivity.py \
  --root "$root" \
  --context "$context" \
  --expected_seeds 0,1,2 \
  --expected_examples_per_run "${PASSKEY_EXAMPLES:-32}" \
  --minimum_source_index "${PASSKEY_SKIP_BASE:-192}" \
  --out_dir "$root/aggregate" \
  --require_complete
