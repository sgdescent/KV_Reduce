#!/bin/bash
set -euo pipefail

dependency="${DEPENDENCY:-}"
root="${ROOT:-outputs/kivi_longbench_instruct_v1}"
examples="${LONGBENCH_EXAMPLES:-24}"
labels=()

submit_model() {
  local model="$1"
  local label="$2"
  local output
  output=$(DEPENDENCY="$dependency" \
    MODEL="$model" MODEL_LABEL="$label" \
    OUT_ROOT="$root/$label" MAX_PROMPT_TOKENS=32752 \
    LONGBENCH_EXAMPLES="$examples" WANDB_PROJECT="${WANDB_PROJECT:-kv-reduce}" \
    bash scripts/submit_longbench_passage_retrieval.sh)
  printf '%s\n' "$output"
  dependency=$(printf '%s\n' "$output" | awk -F= '/longbench_retrieval_aggregate_job=/{print $2}')
  if [[ -z "$dependency" ]]; then
    echo "Failed to parse LongBench aggregate job for $label" >&2
    exit 2
  fi
  labels+=("$label")
}

submit_model Qwen/Qwen2.5-7B-Instruct qwen25_7b_instruct
submit_model meta-llama/Llama-3.1-8B-Instruct llama31_8b_instruct
submit_model Qwen/Qwen3-8B qwen3_8b

models=$(IFS=,; echo "${labels[*]}")
meta_job=$(sbatch --parsable \
  --dependency="afterok:$dependency" \
  --export="ALL,ROOT=$root,MODELS=$models,EXPECTED_EXAMPLES_PER_MODEL=$((3 * examples))" \
  scripts/aggregate_longbench_cross_family.slurm)

printf 'longbench_instruct_final_model_job=%s\n' "$dependency"
printf 'longbench_instruct_meta_job=%s\n' "$meta_job"
printf 'longbench_instruct_output=%s\n' "$root"
