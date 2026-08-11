#!/bin/bash
#SBATCH --job-name=kv_kivi_all_meta
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

root="${ROOT:-outputs/kivi_objective_all_layers_strict_v1}"
out_dir="${OUT_DIR:-$root/cross_family_meta}"
expected_pairs="${EXPECTED_PAIRS:-smollm2_17b_360m,qwen25_3b_15b,olmo2_7b_1b,llama31_8b_32_3b}"

python3 -u aggregate_kivi_cross_family_meta.py \
  --root "$root" \
  --out_dir "$out_dir" \
  --expected_pairs "$expected_pairs" \
  --layout per_pair \
  --require_integrity
