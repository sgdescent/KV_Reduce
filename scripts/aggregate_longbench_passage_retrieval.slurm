#!/bin/bash
#SBATCH --job-name=kv_longbench_agg
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

root="${OUT_ROOT:-outputs/kivi_longbench_passage_retrieval_v1/qwen25_15b}"
python3 -u aggregate_kv_multiple_choice.py \
  --root "$root" \
  --out_dir "$root/aggregate" \
  --expected_tasks longbench_passage_retrieval \
  --expected_seeds 0,1,2 \
  --expected_configs "${EXPECTED_CONFIGS:-none,k8v4,k4v8,k4v4,k4v2,k2v4,k2v2}" \
  --expected_dataset_revision 5e628be450b7e67fb7ae6e201bd6d8f7056f7672 \
  --require_complete
