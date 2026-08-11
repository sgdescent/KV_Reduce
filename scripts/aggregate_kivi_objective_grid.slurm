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
configs="${QUANT_CONFIGS:-none;k16v8;k16v4;k16v3;k16v2;k8v8;k8v4;k8v3;k8v2;k4v8;k4v4;k4v3;k4v2;k3v8;k3v4;k3v3;k3v2;k2v8;k2v4;k2v3;k2v2}"
expected_contexts="${EXPECTED_CONTEXTS:-1024,4096}"
expected_seeds="${EXPECTED_SEEDS:-0,1,2}"
python3 -u aggregate_value_precision_sweep.py \
  --sweep_dir "$root/spec" \
  --out_dir "$root/spec_aggregate" \
  --expected_contexts "$expected_contexts" \
  --expected_seeds "$expected_seeds" \
  --expected_configs "$configs" \
  --require_complete \
  --require_full_runs \
  --require_exact_target
python3 -u aggregate_value_precision_quality_sweep.py \
  --sweep_dir "$root/quality" \
  --out_dir "$root/quality_aggregate" \
  --require_full_runs
python3 -u compare_value_precision_objectives.py \
  --spec_summary "$root/spec_aggregate/summary.json" \
  --quality_summary "$root/quality_aggregate/summary.json" \
  --out_dir "$root/objective_comparison"
python3 -u aggregate_kivi_objective_preferences.py \
  --spec_dir "$root/spec" \
  --quality_dir "$root/quality" \
  --out_dir "$root/objective_comparison"
