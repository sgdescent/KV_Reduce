#!/bin/bash
#SBATCH --job-name=kv_joint_select
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

joint_root="${JOINT_ROOT:-outputs/kivi_joint_target_draft/qwen25_3b_15b}"
quality_root="${QUALITY_ROOT:-outputs/kivi_standalone_quality/qwen25_3b}"
python3 -u select_joint_kv_quantization_policy.py \
  --joint_summary "$joint_root/aggregate/summary.json" \
  --target_quality_summary "$quality_root/aggregate/summary.json" \
  --out_dir "$joint_root/policy_selection" \
  --target_kl_max "${TARGET_KL_MAX:-0.01}" \
  --target_delta_nll_max "${TARGET_DELTA_NLL_MAX:-0.02}" \
  --target_top1_min "${TARGET_TOP1_MIN:-0.95}" \
  --acceptance_drop_max "${ACCEPTANCE_DROP_MAX:-0.02}" \
  --target_token_match_drop_max "${TARGET_TOKEN_MATCH_DROP_MAX:-0.03}" \
  --target_sequence_match_drop_max "${TARGET_SEQUENCE_MATCH_DROP_MAX:-0.10}"
