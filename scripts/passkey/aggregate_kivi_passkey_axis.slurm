#!/bin/bash
#SBATCH --job-name=kv_passkey_axis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

python3 -u -m kv_reduce.analysis.aggregate_kv_passkey_axis \
  --per_channel_root "${PER_CHANNEL_ROOT:-outputs/kivi_passkey_confusable_v3/qwen25_15b}" \
  --per_token_root "${PER_TOKEN_ROOT:-outputs/kivi_passkey_axis_v1/qwen25_15b_per_token}" \
  --out_dir "${OUT_DIR:-outputs/kivi_passkey_axis_v1/qwen25_15b_comparison}" \
  --context "${PASSKEY_CONTEXT:-16384}" \
  --expected_examples_per_run "${PASSKEY_EXAMPLES:-16}" \
  --expected_num_choices "${PASSKEY_NUM_CHOICES:-16}" \
  --expected_generator_version "${PASSKEY_GENERATOR_VERSION:-synthetic_associative_passkey_v3}" \
  --expected_passkey_variant "${PASSKEY_VARIANT:-confusable_records}" \
  --expected_passkey_score "${PASSKEY_SCORE:-normalized}" \
  --require_complete
