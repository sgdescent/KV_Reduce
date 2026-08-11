#!/bin/bash
#SBATCH --job-name=kv_passkey_axis
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --exclude=catalyst-0-[9,15]
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "${REPO_ROOT:-/home/sakshamg/KV_Reduce}"
if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

python3 -u aggregate_kv_passkey_axis.py \
  --per_channel_root "${PER_CHANNEL_ROOT:-outputs/kivi_passkey_confusable_v3/qwen25_15b}" \
  --per_token_root "${PER_TOKEN_ROOT:-outputs/kivi_passkey_axis_v1/qwen25_15b_per_token}" \
  --out_dir "${OUT_DIR:-outputs/kivi_passkey_axis_v1/qwen25_15b_comparison}" \
  --context "${PASSKEY_CONTEXT:-16384}" \
  --expected_examples_per_run "${PASSKEY_EXAMPLES:-16}" \
  --require_complete
