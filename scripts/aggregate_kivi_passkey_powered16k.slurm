#!/bin/bash
#SBATCH --job-name=kv_passkey_pmeta
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

root="${ROOT:-outputs/kivi_passkey_confusable_powered16k_v1}"
python3 -u aggregate_kv_passkey_cross_family.py \
  --source "Qwen2.5-1.5B=$root/qwen25_15b/aggregate/summary.json" \
  --source "Llama-3.2-3B=$root/llama32_3b/aggregate/summary.json" \
  --source "Qwen3-4B=$root/qwen3_4b/aggregate/summary.json" \
  --out_dir "$root/cross_family" \
  --expected_contexts 16384 \
  --expected_seeds 0,1,2 \
  --expected_examples_per_run "${PASSKEY_EXAMPLES:-64}"
