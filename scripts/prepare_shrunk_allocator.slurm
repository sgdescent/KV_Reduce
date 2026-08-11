#!/bin/bash
#SBATCH --job-name=kv_shrink_prep
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "${REPO_ROOT:-/home/sakshamg/KV_Reduce}"
source .venv/bin/activate

python3 -u prepare_shrunk_kv_allocations.py \
  --quality_profile_csv "${QUALITY_PROFILE_CSV:?Set QUALITY_PROFILE_CSV}" \
  --acceptance_profile_csv "${ACCEPTANCE_PROFILE_CSV:?Set ACCEPTANCE_PROFILE_CSV}" \
  --quality_allocation "${QUALITY_ALLOCATION:?Set QUALITY_ALLOCATION}" \
  --acceptance_allocation "${ACCEPTANCE_ALLOCATION:?Set ACCEPTANCE_ALLOCATION}" \
  --num_layers "${NUM_LAYERS:?Set NUM_LAYERS}" \
  --target_profiled_mean_bits "${TARGET_PROFILED_MEAN_BITS:-8}" \
  --prior_strengths "${PRIOR_STRENGTHS:-0.25,1,4}" \
  --include_ucb \
  --out_dir "${OUT_ROOT:?Set OUT_ROOT}"

