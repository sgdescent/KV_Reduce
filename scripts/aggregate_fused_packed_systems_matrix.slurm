#!/bin/bash
#SBATCH --job-name=kv_pack_matagg
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

root="${OUT_ROOT:-outputs/fused_packed_systems_matrix_v1}"
python3 -u aggregate_fused_packed_systems.py \
  --root "$root" \
  --out_dir "$root/aggregate" \
  --require_complete
