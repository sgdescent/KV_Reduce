#!/bin/bash
#SBATCH --job-name=kv_keyresidagg
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --exclude=catalyst-0-[9,15]
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
cd "${REPO_ROOT:-/home/sakshamg/KV_Reduce}"
if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

residual_lengths=(0 32 128 256)
residual_length="${residual_lengths[${SLURM_ARRAY_TASK_ID:-0}]}"
root="${OUT_ROOT:-outputs/key_residual_geometry_v1/qwen25_3b_15b}"

python3 -u aggregate_sequential_spec_runs.py \
  --summary_glob "$root/residual_${residual_length}/seed_*/summary.json" \
  --out_dir "$root/residual_${residual_length}/aggregate" \
  --bootstrap_samples "${BOOTSTRAP_SAMPLES:-30000}" \
  --require_full_runs
