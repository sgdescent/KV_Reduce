#!/bin/bash
#SBATCH --job-name=kv_passkey_agga
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

root="${OUT_ROOT:-outputs/kivi_passkey_aggressive_v1/qwen25_15b}"
python3 -u aggregate_kv_passkey.py \
  --root "$root" \
  --out_dir "$root/aggregate" \
  --expected_contexts 8192,16384,32768 \
  --expected_seeds 0,1,2 \
  --expected_examples_per_run "${PASSKEY_EXAMPLES:-16}" \
  --require_complete
