#!/bin/bash
#SBATCH --job-name=kv_keyresidmeta
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

root="${OUT_ROOT:-outputs/key_residual_geometry_v2_paired/qwen25_3b_15b}"
python3 -u -m kv_reduce.analysis.aggregate_paired_geometry_sweep \
  --root "$root" \
  --out_dir "$root/aggregate" \
  --treatment_prefix residual \
  --treatment_config_key key_residual_length \
  --treatment_label "BF16 residual tokens" \
  --treatments 0,32,128,256 \
  --reference 128 \
  --expected_seeds 0,1,2 \
  --expected_prompts "${NUM_PROMPTS:-32}"
