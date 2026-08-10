#!/bin/bash
#SBATCH --job-name=kv_kivi_obj_agg
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

root="${ROOT:-outputs/kivi_objective_grid/qwen25_3b_15b}"
python3 -u aggregate_value_precision_sweep.py \
  --sweep_dir "$root/spec" \
  --out_dir "$root/spec_aggregate"
python3 -u aggregate_value_precision_quality_sweep.py \
  --sweep_dir "$root/quality" \
  --out_dir "$root/quality_aggregate"
python3 -u compare_value_precision_objectives.py \
  --spec_summary "$root/spec_aggregate/summary.json" \
  --quality_summary "$root/quality_aggregate/summary.json" \
  --out_dir "$root/objective_comparison"
