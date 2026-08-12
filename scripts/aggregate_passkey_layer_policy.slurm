#!/bin/bash
#SBATCH --job-name=kv_passkey_pagg
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
if [[ -f .venv/bin/activate ]]; then source .venv/bin/activate; fi
task_id=$((10#${SLURM_ARRAY_TASK_ID:-0}))
case "$task_id" in
  0) model_tag="qwen25_15b" ;;
  1) model_tag="llama32_3b" ;;
  2) model_tag="qwen3_4b" ;;
  *) exit 2 ;;
esac
context="${PASSKEY_CONTEXT:-16384}"
root="${OUT_ROOT:-outputs/passkey_layer_policy_v1}/$model_tag"
python3 -u aggregate_kv_passkey.py \
  --root "$root" --out_dir "$root/aggregate" \
  --expected_contexts "$context" --expected_seeds 0,1,2 \
  --expected_examples_per_run "${PASSKEY_EXAMPLES:-32}" \
  --expected_num_choices 16 \
  --expected_generator_version synthetic_associative_passkey_v3 \
  --expected_passkey_variant confusable_records --expected_passkey_score raw \
  --minimum_source_index "${PASSKEY_SKIP_BASE:-288}" \
  --comparison_pairs 'retrieval_aware_mixed:all_k2v4;retrieval_aware_mixed:all_k4v2;retrieval_aware_mixed:selected_k4v4' \
  --require_complete
