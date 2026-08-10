#!/bin/bash
#SBATCH --job-name=kv_kivi_mcmeta
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

root="${OUT_BASE:-outputs/kivi_multiple_choice_cross_family}"
qwen_summary="${QWEN_SUMMARY:-outputs/kivi_multiple_choice/qwen25_15b/aggregate/summary.json}"
python3 -u aggregate_kv_multiple_choice_cross_family.py \
  --root "$root" \
  --qwen_summary "$qwen_summary" \
  --out_dir "$root/aggregate"
