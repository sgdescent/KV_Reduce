#!/bin/bash
#SBATCH --job-name=kv_longbench_meta
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

root="${ROOT:-outputs/kivi_longbench_passage_retrieval_v1}"
python3 -u -m kv_reduce.analysis.aggregate_longbench_cross_family \
  --root "$root" \
  --out_dir "$root/cross_family_aggregate" \
  --models "${MODELS:-qwen25_15b,llama32_3b,qwen3_4b}" \
  --expected_examples_per_model "${EXPECTED_EXAMPLES_PER_MODEL:-72}"
