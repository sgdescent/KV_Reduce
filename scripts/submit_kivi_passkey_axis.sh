#!/bin/bash
set -euo pipefail

dependency="${DEPENDENCY:-}"
model="${MODEL:-Qwen/Qwen2.5-1.5B}"
model_tag="${MODEL_TAG:-qwen25_15b}"
examples="${PASSKEY_EXAMPLES:-16}"
per_channel_root="${PER_CHANNEL_ROOT:-outputs/kivi_passkey_confusable_v3/$model_tag}"
per_token_root="${PER_TOKEN_ROOT:-outputs/kivi_passkey_axis_v1/${model_tag}_per_token}"
comparison_root="${COMPARISON_ROOT:-outputs/kivi_passkey_axis_v1/${model_tag}_comparison}"

submission=$(
  DEPENDENCY="$dependency" \
  MODEL="$model" \
  MODEL_TAG="$model_tag" \
  OUT_ROOT="$per_token_root" \
  PASSKEY_CONTEXTS=16384 \
  PASSKEY_EXAMPLES="$examples" \
  KEY_QUANT_AXIS=per_token \
  KEY_GROUP_SIZE=32 \
  KEY_RESIDUAL_LENGTH=128 \
  VALUE_QUANT_SCHEME=affine \
  PASSKEY_WANDB_GROUP=kivi-passkey-axis-per-token \
  bash scripts/submit_kivi_passkey_confusable.sh
)
printf '%s\n' "$submission"
per_token_aggregate=$(
  printf '%s\n' "$submission" \
    | awk -F= '/passkey_aggressive_aggregate_job=/{print $2}'
)
if [[ -z "$per_token_aggregate" ]]; then
  echo "Failed to parse per-token passkey aggregate job" >&2
  exit 2
fi

axis_job=$(sbatch --parsable \
  --dependency="afterok:${per_token_aggregate}" \
  --exclude=catalyst-0-9,catalyst-0-15 \
  --export="ALL,PER_CHANNEL_ROOT=$per_channel_root,PER_TOKEN_ROOT=$per_token_root,OUT_DIR=$comparison_root,PASSKEY_CONTEXT=16384,PASSKEY_EXAMPLES=$examples" \
  scripts/aggregate_kivi_passkey_axis.slurm)

printf 'passkey_axis_aggregate_job=%s\n' "$axis_job"
printf 'passkey_axis_output=%s\n' "$comparison_root"
