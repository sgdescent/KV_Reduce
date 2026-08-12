#!/bin/bash
#SBATCH --job-name=kv_passkey_axagg
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
if [[ -f .venv/bin/activate ]]; then source .venv/bin/activate; fi

root="${OUT_ROOT:-outputs/kivi_passkey_axis_powered16k_v1/qwen25_15b_per_token}"
python3 -u aggregate_kv_passkey.py \
  --root "$root" --out_dir "$root/aggregate" \
  --expected_contexts "${PASSKEY_CONTEXT:-16384}" \
  --expected_seeds 0,1,2 \
  --expected_examples_per_run "${PASSKEY_EXAMPLES:-64}" \
  --expected_num_choices 16 \
  --expected_generator_version synthetic_associative_passkey_v3 \
  --expected_passkey_variant confusable_records \
  --expected_passkey_score raw \
  --require_complete
