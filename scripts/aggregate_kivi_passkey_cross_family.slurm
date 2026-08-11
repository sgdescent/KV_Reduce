#!/bin/bash
#SBATCH --job-name=kv_passkey_meta
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "${REPO_ROOT:-/home/sakshamg/KV_Reduce}"
if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

root="${ROOT:-outputs/kivi_passkey_confusable_v3}"
python3 -u aggregate_kv_passkey_cross_family.py \
  --source "qwen25_15b=$root/qwen25_15b/aggregate/summary.json" \
  --source "llama32_3b=$root/llama32_3b/aggregate/summary.json" \
  --source "qwen3_4b=$root/qwen3_4b/aggregate/summary.json" \
  --out_dir "$root/cross_family_meta" \
  --expected_contexts 8192,16384,32768 \
  --expected_seeds 0,1,2
