#!/bin/bash
set -euo pipefail

dependency="${DEPENDENCY:-}"

submit_model() {
  local model="$1"
  local label="$2"
  local output
  output=$(DEPENDENCY="$dependency" \
    MODEL="$model" \
    MODEL_LABEL="$label" \
    OUT_ROOT="outputs/kivi_longbench_passage_retrieval_v1/$label" \
    MAX_PROMPT_TOKENS=32752 \
    WANDB_PROJECT="${WANDB_PROJECT:-kv-reduce}" \
    bash scripts/submit_longbench_passage_retrieval.sh)
  printf '%s\n' "$output"
  dependency=$(printf '%s\n' "$output" | awk -F= '/longbench_retrieval_aggregate_job=/{print $2}')
  if [[ -z "$dependency" ]]; then
    echo "Failed to parse LongBench aggregate job for $label" >&2
    exit 2
  fi
}

submit_model meta-llama/Llama-3.2-3B llama32_3b
submit_model Qwen/Qwen3-4B qwen3_4b

printf 'longbench_cross_family_final_job=%s\n' "$dependency"
