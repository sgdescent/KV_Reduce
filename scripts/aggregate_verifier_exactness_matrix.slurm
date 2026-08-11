#!/bin/bash
#SBATCH --job-name=kv_verify_agg
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "${REPO_ROOT:-/home/sakshamg/KV_Reduce}"
if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

root="${OUT_ROOT:-outputs/target_verification_diagnostic/model_matrix}"
python3 -u aggregate_verifier_exactness_matrix.py \
  --root "$root/raw" \
  --tie_margin "${TIE_MARGIN:-0.001}" \
  --out_dir "$root"
