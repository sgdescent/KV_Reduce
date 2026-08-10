#!/bin/bash
set -euo pipefail

dependency="${DEPENDENCY:-}"
num_seeds="${NUM_SEEDS:-2}"
examples="${NUM_EXAMPLES:-128}"
out_base="${OUT_BASE:-outputs/kivi_multiple_choice_cross_family}"

if (( num_seeds <= 0 || examples <= 0 )); then
  echo "NUM_SEEDS and NUM_EXAMPLES must be positive" >&2
  exit 2
fi

labels=(llama32_3b olmo2_1b smollm2_360m)
models=(
  meta-llama/Llama-3.2-3B
  allenai/OLMo-2-0425-1B
  HuggingFaceTB/SmolLM2-360M
)

previous="$dependency"
array_max=$((2 * num_seeds - 1))
for index in "${!labels[@]}"; do
  label="${labels[$index]}"
  model="${models[$index]}"
  root="$out_base/$label"
  dependency_args=()
  if [[ -n "$previous" ]]; then
    dependency_args+=(--dependency="afterok:${previous}")
  fi

  array_job=$(sbatch --parsable \
    "${dependency_args[@]}" \
    --array="0-${array_max}%1" \
    --exclude=catalyst-0-9,catalyst-0-15 \
    --export="ALL,MODEL=${model},MODEL_LABEL=${label},NUM_SEEDS=${num_seeds},HELLASWAG_EXAMPLES=${examples},ARC_EXAMPLES=${examples},OUT_ROOT=${root},WANDB_PROJECT=${WANDB_PROJECT:-kv-reduce},WANDB_GROUP=kivi-multiple-choice-cross-family" \
    scripts/run_kivi_multiple_choice.slurm)
  aggregate_job=$(sbatch --parsable \
    --dependency="afterok:${array_job}" \
    --exclude=catalyst-0-9,catalyst-0-15 \
    --export="ALL,OUT_ROOT=${root}" \
    scripts/aggregate_kivi_multiple_choice.slurm)

  printf '%s_array_job=%s\n' "$label" "$array_job"
  printf '%s_aggregate_job=%s\n' "$label" "$aggregate_job"
  previous="$array_job"
done
