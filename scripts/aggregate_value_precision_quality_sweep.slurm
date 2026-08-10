#!/bin/bash
#SBATCH --job-name=kv_value_quality_agg
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

python3 -u aggregate_value_precision_quality_sweep.py \
  --sweep_dir "${SWEEP_ROOT:-outputs/value_precision_quality/qwen25_15b}" \
  --out_dir "${OUT_DIR:-outputs/value_precision_quality/qwen25_15b/aggregate}"

spec_summary="${SPEC_SUMMARY:-outputs/value_precision_sweep/qwen25_3b_15b/aggregate/summary.json}"
if [[ -f "$spec_summary" ]]; then
  python3 -u compare_value_precision_objectives.py \
    --spec_summary "$spec_summary" \
    --quality_summary "${OUT_DIR:-outputs/value_precision_quality/qwen25_15b/aggregate}/summary.json" \
    --out_dir "${COMPARISON_OUT:-outputs/value_precision_objectives/qwen25_3b_15b}"
fi
