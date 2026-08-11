#!/bin/bash
set -euo pipefail

dependency="${DEPENDENCY:-}"
out_root="${OUT_ROOT:-outputs/kivi_passkey_aggressive_v2/qwen25_15b}"
num_choices="${PASSKEY_NUM_CHOICES:-16}"
variant="${PASSKEY_VARIANT:-random}"
if [[ "$variant" == "confusable_records" ]]; then
  default_generator_version="synthetic_associative_passkey_v3"
else
  default_generator_version="synthetic_passkey_16way_v2"
fi
generator_version="${PASSKEY_GENERATOR_VERSION:-$default_generator_version}"
dependency_args=()
if [[ -n "$dependency" ]]; then
  dependency_args+=(--dependency="afterok:${dependency}")
fi

array_job=$(sbatch --parsable \
  "${dependency_args[@]}" \
  --array=0-8%1 \
  --exclude=catalyst-0-9,catalyst-0-15 \
  --export="ALL,WANDB_PROJECT=${WANDB_PROJECT:-kv-reduce},PASSKEY_EXAMPLES=${PASSKEY_EXAMPLES:-16},PASSKEY_NUM_CHOICES=$num_choices,PASSKEY_VARIANT=$variant,PASSKEY_WANDB_GROUP=${PASSKEY_WANDB_GROUP:-kivi-passkey-aggressive},OUT_ROOT=$out_root" \
  scripts/run_kivi_passkey_aggressive.slurm)
aggregate_job=$(sbatch --parsable \
  --dependency="afterok:${array_job}" \
  --exclude=catalyst-0-9,catalyst-0-15 \
  --export="ALL,PASSKEY_EXAMPLES=${PASSKEY_EXAMPLES:-16},PASSKEY_NUM_CHOICES=$num_choices,PASSKEY_VARIANT=$variant,PASSKEY_GENERATOR_VERSION=$generator_version,OUT_ROOT=$out_root" \
  scripts/aggregate_kivi_passkey_aggressive.slurm)

printf 'passkey_aggressive_job=%s\n' "$array_job"
printf 'passkey_aggressive_aggregate_job=%s\n' "$aggregate_job"
printf 'passkey_aggressive_output=%s\n' "$out_root"
