#!/bin/bash
#SBATCH --job-name=kv_kivi_ragg
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
cd "${REPO_ROOT:-/home/sakshamg/KV_Reduce}"
if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

domains=(c4 gsm8k humaneval)
domain="${domains[${SLURM_ARRAY_TASK_ID:-0}]}"
root="${ROOT:-outputs/kivi_objective_robustness}"

python3 -u aggregate_value_precision_sweep.py \
  --sweep_dir "$root/spec/$domain" \
  --out_dir "$root/spec/$domain/aggregate"
python3 -u aggregate_value_precision_quality_sweep.py \
  --sweep_dir "$root/quality/$domain" \
  --out_dir "$root/quality/$domain/aggregate"
python3 -u compare_value_precision_objectives.py \
  --spec_summary "$root/spec/$domain/aggregate/summary.json" \
  --quality_summary "$root/quality/$domain/aggregate/summary.json" \
  --out_dir "$root/comparison/$domain"
python3 -u aggregate_kivi_objective_preferences.py \
  --spec_dir "$root/spec/$domain" \
  --quality_dir "$root/quality/$domain" \
  --config_pairs "k8v4,k4v8;k4v3,k3v4;k4v2,k2v4" \
  --out_dir "$root/comparison/$domain"
