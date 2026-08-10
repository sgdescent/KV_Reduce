#!/bin/bash
set -euo pipefail

dependency="${DEPENDENCY:-}"
num_seeds="${NUM_SEEDS:-2}"
hellaswag_examples="${HELLASWAG_EXAMPLES:-${NUM_EXAMPLES:-128}}"
arc_validation_size="${ARC_VALIDATION_SIZE:-299}"
arc_examples="${ARC_EXAMPLES:-$((arc_validation_size / num_seeds))}"
out_base="${OUT_BASE:-outputs/kivi_multiple_choice_cross_family}"

if (( num_seeds <= 0 || hellaswag_examples <= 0 || arc_examples <= 0 )); then
  echo "NUM_SEEDS, HELLASWAG_EXAMPLES, and ARC_EXAMPLES must be positive" >&2
  exit 2
fi
if (( num_seeds * arc_examples > arc_validation_size )); then
  echo "ARC shards request $((num_seeds * arc_examples)) examples, exceeding $arc_validation_size" >&2
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
aggregate_jobs=()
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
    --export="ALL,MODEL=${model},MODEL_LABEL=${label},NUM_SEEDS=${num_seeds},HELLASWAG_EXAMPLES=${hellaswag_examples},ARC_EXAMPLES=${arc_examples},ARC_VALIDATION_SIZE=${arc_validation_size},OUT_ROOT=${root},WANDB_PROJECT=${WANDB_PROJECT:-kv-reduce},WANDB_GROUP=${WANDB_GROUP:-kivi-multiple-choice-cross-family}" \
    scripts/run_kivi_multiple_choice.slurm)
  aggregate_job=$(sbatch --parsable \
    --dependency="afterok:${array_job}" \
    --exclude=catalyst-0-9,catalyst-0-15 \
    --export="ALL,OUT_ROOT=${root}" \
    scripts/aggregate_kivi_multiple_choice.slurm)

  printf '%s_array_job=%s\n' "$label" "$array_job"
  printf '%s_aggregate_job=%s\n' "$label" "$aggregate_job"
  aggregate_jobs+=("$aggregate_job")
  previous="$array_job"
done

aggregate_dependency=$(IFS=:; printf '%s' "${aggregate_jobs[*]}")
meta_job=$(sbatch --parsable \
  --dependency="afterok:${aggregate_dependency}" \
  --exclude=catalyst-0-9,catalyst-0-15 \
  --export="ALL,OUT_BASE=${out_base}" \
  scripts/aggregate_kivi_multiple_choice_cross_family.slurm)
printf 'cross_family_meta_job=%s\n' "$meta_job"
