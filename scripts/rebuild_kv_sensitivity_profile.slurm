#!/bin/bash
#SBATCH --job-name=kv_profile_clean
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs outputs

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
PROFILE_CSV="${PROFILE_CSV:?Set PROFILE_CSV to the source profile summary}"
RAW_CSV="${RAW_CSV:?Set RAW_CSV to the source per-example rows}"
OBJECTIVE="${OBJECTIVE:?Set OBJECTIVE to quality or acceptance}"
NUM_SAMPLES="${NUM_SAMPLES:?Set NUM_SAMPLES to the requested profile size}"
OUT_DIR="${OUT_DIR:?Set OUT_DIR for the rebuilt profile}"
QUALITY_RISK_METRIC="${QUALITY_RISK_METRIC:-kl}"
EXACTNESS_TIE_MARGIN="${EXACTNESS_TIE_MARGIN:-0.001}"
Z_SCORE="${Z_SCORE:-1.96}"

"$PYTHON_BIN" -u resample_kv_sensitivity_profile.py \
  --profile_csv "$PROFILE_CSV" \
  --raw_csv "$RAW_CSV" \
  --objective "$OBJECTIVE" \
  --num_samples "$NUM_SAMPLES" \
  --quality_risk_metric "$QUALITY_RISK_METRIC" \
  --exactness_tie_margin "$EXACTNESS_TIE_MARGIN" \
  --z_score "$Z_SCORE" \
  --out_dir "$OUT_DIR"
