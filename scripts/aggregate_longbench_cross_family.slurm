#!/bin/bash
#SBATCH --job-name=kv_longbench_meta
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

root="${ROOT:-outputs/kivi_longbench_passage_retrieval_v1}"
python3 -u aggregate_longbench_cross_family.py \
  --root "$root" \
  --out_dir "$root/cross_family_aggregate" \
  --models qwen25_15b,llama32_3b,qwen3_4b \
  --expected_examples_per_model 72
