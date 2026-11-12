#!/bin/bash
#SBATCH --job-name=kv_kivi_xmeta
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

root="${ROOT:-outputs/kivi_objective_cross_family}"
python3 -u -m kv_reduce.analysis.aggregate_kivi_cross_family_meta \
  --root "$root" \
  --out_dir "$root/meta_aggregate" \
  --acceptance_drop_budget "${ACCEPTANCE_DROP_BUDGET:-0.02}" \
  --quality_kl_budget "${QUALITY_KL_BUDGET:-0.01}"
