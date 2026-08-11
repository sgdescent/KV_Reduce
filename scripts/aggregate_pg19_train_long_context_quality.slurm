#!/bin/bash
#SBATCH --job-name=kv_qual_pgagg
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

root="${OUT_ROOT:-outputs/quality_long_context_v2_pg19_train/qwen25_15b}"
python3 -u aggregate_value_precision_quality_sweep.py \
  --sweep_dir "$root" \
  --out_dir "$root/aggregate" \
  --require_full_runs
