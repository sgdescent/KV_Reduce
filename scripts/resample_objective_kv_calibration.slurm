#!/bin/bash
#SBATCH --job-name=kv_calib_ablate
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
mkdir -p logs outputs

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
SOURCE_ROOT="${SOURCE_ROOT:?Set SOURCE_ROOT to a completed objective campaign root}"
OUT_DIR="${OUT_DIR:-$SOURCE_ROOT/calibration_ablation}"
NUM_LAYERS="${NUM_LAYERS:-28}"
SAMPLE_SIZES="${SAMPLE_SIZES:-4;8;16}"
ALLOWED_BITS="${ALLOWED_BITS:-4;8;16}"
TARGET_PROFILED_MEAN_BITS="${TARGET_PROFILED_MEAN_BITS:-8}"
QUALITY_RISK_FIELD="${QUALITY_RISK_FIELD:-quality_risk}"
ACCEPTANCE_RISK_FIELD="${ACCEPTANCE_RISK_FIELD:-accept_rate_drop_ucb95_clipped}"

QUALITY_PROFILE="$SOURCE_ROOT/quality_profile/profile_summary.csv"
QUALITY_RAW="$SOURCE_ROOT/quality_profile/raw_sequence_rows.csv"
ACCEPTANCE_PROFILE="$SOURCE_ROOT/acceptance_profile/profile_summary.csv"
ACCEPTANCE_RAW="$SOURCE_ROOT/acceptance_profile/raw_prompt_rows.csv"

for required in "$QUALITY_PROFILE" "$QUALITY_RAW" "$ACCEPTANCE_PROFILE" "$ACCEPTANCE_RAW"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing required profile artifact: $required" >&2
    exit 1
  fi
done

IFS=';' read -r -a sizes <<< "$SAMPLE_SIZES"
if [[ "${#sizes[@]}" -lt 2 ]]; then
  echo "SAMPLE_SIZES must contain at least two semicolon-delimited sizes." >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
acceptance_profile_args=()
for sample_size in "${sizes[@]}"; do
  sample_root="$OUT_DIR/n${sample_size}"
  quality_out="$sample_root/quality_profile"
  acceptance_out="$sample_root/acceptance_profile"
  quality_allocation="$sample_root/quality_allocation"
  acceptance_allocation="$sample_root/acceptance_allocation"
  comparison_out="$sample_root/objective_comparison"

  "$PYTHON_BIN" -u resample_kv_sensitivity_profile.py \
    --profile_csv "$QUALITY_PROFILE" \
    --raw_csv "$QUALITY_RAW" \
    --objective quality \
    --num_samples "$sample_size" \
    --out_dir "$quality_out"

  "$PYTHON_BIN" -u resample_kv_sensitivity_profile.py \
    --profile_csv "$ACCEPTANCE_PROFILE" \
    --raw_csv "$ACCEPTANCE_RAW" \
    --objective acceptance \
    --num_samples "$sample_size" \
    --out_dir "$acceptance_out"

  "$PYTHON_BIN" -u search_kv_bit_allocation.py \
    --profile_csv "$quality_out/profile_summary.csv" \
    --num_layers "$NUM_LAYERS" \
    --allowed_bits "$ALLOWED_BITS" \
    --risk_field "$QUALITY_RISK_FIELD" \
    --target_profiled_mean_bits "$TARGET_PROFILED_MEAN_BITS" \
    --name "quality_n${sample_size}" \
    --out_dir "$quality_allocation"

  "$PYTHON_BIN" -u search_kv_bit_allocation.py \
    --profile_csv "$acceptance_out/profile_summary.csv" \
    --num_layers "$NUM_LAYERS" \
    --allowed_bits "$ALLOWED_BITS" \
    --risk_field "$ACCEPTANCE_RISK_FIELD" \
    --target_profiled_mean_bits "$TARGET_PROFILED_MEAN_BITS" \
    --name "acceptance_n${sample_size}" \
    --out_dir "$acceptance_allocation"

  "$PYTHON_BIN" -u compare_kv_objectives.py \
    --quality_profile_csv "$quality_out/profile_summary.csv" \
    --acceptance_profile_csv "$acceptance_out/profile_summary.csv" \
    --quality_allocation "$quality_allocation/allocation.json" \
    --acceptance_allocation "$acceptance_allocation/allocation.json" \
    --out_dir "$comparison_out"

  acceptance_profile_args+=(
    --profile "n${sample_size}=$acceptance_out/profile_summary.csv"
  )
done

"$PYTHON_BIN" -u paper/compare_calibration_profiles.py \
  "${acceptance_profile_args[@]}" \
  --out_dir "$OUT_DIR/acceptance_stability"

echo "Calibration-size ablation written to $OUT_DIR"
