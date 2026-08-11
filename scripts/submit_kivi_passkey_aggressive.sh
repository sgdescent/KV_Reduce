#!/bin/bash
set -euo pipefail

dependency="${DEPENDENCY:-}"
model="${MODEL:-Qwen/Qwen2.5-1.5B}"
model_tag="${MODEL_TAG:-qwen25_15b}"
out_root="${OUT_ROOT:-outputs/kivi_passkey_aggressive_v2/$model_tag}"
num_choices="${PASSKEY_NUM_CHOICES:-16}"
variant="${PASSKEY_VARIANT:-random}"
score="${PASSKEY_SCORE:-raw}"
if [[ "$variant" == "confusable_records" ]]; then
  default_generator_version="synthetic_associative_passkey_v3"
else
  default_generator_version="synthetic_passkey_16way_v2"
fi
generator_version="${PASSKEY_GENERATOR_VERSION:-$default_generator_version}"
contexts="${PASSKEY_CONTEXTS:-8192,16384,32768}"
IFS=',' read -r -a context_values <<< "$contexts"
array_end=$((3 * ${#context_values[@]} - 1))
dependency_arg=""
if [[ -n "$dependency" ]]; then
  dependency_arg="--dependency=afterok:${dependency}"
fi

array_job=$(sbatch --parsable \
  ${dependency_arg:+"$dependency_arg"} \
  --array="0-${array_end}%1" \
  --exclude=catalyst-0-9,catalyst-0-15 \
  --export="ALL,WANDB_PROJECT=${WANDB_PROJECT:-kv-reduce},MODEL=$model,MODEL_TAG=$model_tag,PASSKEY_CONTEXTS=$contexts,PASSKEY_EXAMPLES=${PASSKEY_EXAMPLES:-16},PASSKEY_NUM_CHOICES=$num_choices,PASSKEY_VARIANT=$variant,PASSKEY_SCORE=$score,PASSKEY_WANDB_GROUP=${PASSKEY_WANDB_GROUP:-kivi-passkey-aggressive},KEY_QUANT_AXIS=${KEY_QUANT_AXIS:-per_channel},KEY_GROUP_SIZE=${KEY_GROUP_SIZE:-32},KEY_RESIDUAL_LENGTH=${KEY_RESIDUAL_LENGTH:-128},VALUE_QUANT_SCHEME=${VALUE_QUANT_SCHEME:-affine},OUT_ROOT=$out_root" \
  scripts/run_kivi_passkey_aggressive.slurm)
aggregate_job=$(sbatch --parsable \
  --dependency="afterok:${array_job}" \
  --exclude=catalyst-0-9,catalyst-0-15 \
  --export="ALL,PASSKEY_CONTEXTS=$contexts,PASSKEY_EXAMPLES=${PASSKEY_EXAMPLES:-16},PASSKEY_NUM_CHOICES=$num_choices,PASSKEY_VARIANT=$variant,PASSKEY_SCORE=$score,PASSKEY_GENERATOR_VERSION=$generator_version,OUT_ROOT=$out_root" \
  scripts/aggregate_kivi_passkey_aggressive.slurm)

printf 'passkey_aggressive_job=%s\n' "$array_job"
printf 'passkey_aggressive_aggregate_job=%s\n' "$aggregate_job"
printf 'passkey_aggressive_output=%s\n' "$out_root"
