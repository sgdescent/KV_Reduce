#!/bin/bash
#SBATCH --job-name=kv_pg19_objcmp
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

python3 -u compare_sequential_quality_objectives.py \
  --spec_summary "${SPEC_SUMMARY:-outputs/sequential_long_context_v2_pg19_train/qwen25_3b_15b/aggregate/summary.json}" \
  --quality_summary "${QUALITY_SUMMARY:-outputs/quality_long_context_v2_pg19_train/qwen25_15b/aggregate/summary.json}" \
  --out_dir "${OUT_DIR:-outputs/pg19_train_long_context_objective_comparison/qwen25_3b_15b}"
