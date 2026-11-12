#!/bin/bash
#SBATCH --job-name=kv_keygroupmeta
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

root="${OUT_ROOT:-outputs/key_group_geometry_v2_paired/qwen25_3b_15b}"
python3 -u -m kv_reduce.analysis.aggregate_paired_geometry_sweep \
  --root "$root" \
  --out_dir "$root/aggregate" \
  --treatment_prefix group \
  --treatment_config_key key_group_size \
  --treatment_label "Key group size" \
  --treatments 16,32,64,128 \
  --reference 32 \
  --expected_seeds 0,1,2 \
  --expected_prompts "${NUM_PROMPTS:-32}"
