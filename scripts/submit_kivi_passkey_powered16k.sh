#!/bin/bash
set -euo pipefail

dependency="${DEPENDENCY:-}"
root="${ROOT:-outputs/kivi_passkey_confusable_powered16k_v1}"
examples="${PASSKEY_EXAMPLES:-64}"

submit_model() {
  local model="$1"
  local label="$2"
  local submission
  submission=$(
    DEPENDENCY="$dependency" \
    MODEL="$model" \
    MODEL_TAG="$label" \
    OUT_ROOT="$root/$label" \
    PASSKEY_CONTEXTS=16384 \
    PASSKEY_EXAMPLES="$examples" \
    PASSKEY_WANDB_GROUP=kivi-passkey-powered16k \
    bash scripts/submit_kivi_passkey_confusable.sh
  )
  printf '%s\n' "$submission"
  dependency=$(
    printf '%s\n' "$submission" \
      | awk -F= '/passkey_aggressive_aggregate_job=/{print $2}'
  )
  if [[ -z "$dependency" ]]; then
    echo "Failed to parse powered passkey aggregate job for $label" >&2
    exit 2
  fi
}

submit_model Qwen/Qwen2.5-1.5B qwen25_15b
submit_model meta-llama/Llama-3.2-3B llama32_3b
submit_model Qwen/Qwen3-4B qwen3_4b

meta_job=$(sbatch --parsable \
  --dependency="afterok:${dependency}" \
  --exclude=catalyst-0-9,catalyst-0-15 \
  --export="ALL,ROOT=$root" \
  scripts/aggregate_kivi_passkey_powered16k.slurm)

printf 'passkey_powered16k_final_model_aggregate=%s\n' "$dependency"
printf 'passkey_powered16k_meta_job=%s\n' "$meta_job"
printf 'passkey_powered16k_output=%s\n' "$root"
