#!/bin/bash
#SBATCH --job-name=kv_joint_agg
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

root="${ROOT:-outputs/kivi_joint_target_draft/qwen25_3b_15b}"
python3 -u aggregate_joint_target_draft_quantization.py \
  --sweep_dir "$root" \
  --out_dir "$root/aggregate"
