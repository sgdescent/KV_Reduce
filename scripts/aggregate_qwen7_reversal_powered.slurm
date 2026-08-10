#!/bin/bash
#SBATCH --job-name=kv_q7_rev_agg
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

root="${SWEEP_ROOT:-outputs/kivi_reversal_powered/qwen25_7b_3b}"
.venv/bin/python aggregate_value_precision_sweep.py \
  --sweep_dir "$root/spec" \
  --out_dir "$root/spec_aggregate"
.venv/bin/python aggregate_value_precision_quality_sweep.py \
  --sweep_dir "$root/quality" \
  --out_dir "$root/quality_aggregate"
.venv/bin/python compare_value_precision_objectives.py \
  --spec_summary "$root/spec_aggregate/summary.json" \
  --quality_summary "$root/quality_aggregate/summary.json" \
  --out_dir "$root/objective_comparison"
.venv/bin/python aggregate_kivi_objective_preferences.py \
  --spec_dir "$root/spec" \
  --quality_dir "$root/quality" \
  --config_pairs "k4v3,k3v4" \
  --out_dir "$root/objective_comparison"
