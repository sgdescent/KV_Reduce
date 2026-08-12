#!/bin/bash
set -euo pipefail

dependency="${DEPENDENCY:-}"
per_channel_root="${PER_CHANNEL_ROOT:-outputs/kivi_passkey_confusable_powered16k_v1/qwen25_15b}"
per_token_root="${PER_TOKEN_ROOT:-outputs/kivi_passkey_axis_powered16k_v1/qwen25_15b_per_token}"
comparison_root="${COMPARISON_ROOT:-outputs/kivi_passkey_axis_powered16k_v1/qwen25_15b_comparison}"
dependency_args=()
if [[ -n "$dependency" ]]; then dependency_args+=("--dependency=afterok:$dependency"); fi

eval_job=$(sbatch --parsable --array=0-2%1 \
  "${dependency_args[@]}" \
  --export="ALL,MODEL=Qwen/Qwen2.5-1.5B,MODEL_TAG=qwen25_15b_per_token_powered,OUT_ROOT=$per_token_root,PASSKEY_CONTEXTS=16384,PASSKEY_EXAMPLES=64,PASSKEY_NUM_CHOICES=16,PASSKEY_VARIANT=confusable_records,PASSKEY_SCORE=normalized,QUANT_CONFIGS=none;k4v4;k4v2;k2v4;k2v2,KEY_QUANT_AXIS=per_token,KEY_GROUP_SIZE=32,KEY_RESIDUAL_LENGTH=0,VALUE_QUANT_SCHEME=affine,WANDB_PROJECT=kv-reduce,PASSKEY_WANDB_GROUP=passkey-axis-powered-per-token" \
  scripts/run_kivi_passkey_aggressive.slurm)

input_aggregate=$(sbatch --parsable \
  --dependency="afterok:$eval_job" \
  --export="ALL,OUT_ROOT=$per_token_root,PASSKEY_CONTEXT=16384,PASSKEY_EXAMPLES=64" \
  scripts/aggregate_powered_passkey_axis_input.slurm)

axis_aggregate=$(sbatch --parsable \
  --dependency="afterok:$input_aggregate" \
  --export="ALL,PER_CHANNEL_ROOT=$per_channel_root,PER_TOKEN_ROOT=$per_token_root,OUT_DIR=$comparison_root,PASSKEY_CONTEXT=16384,PASSKEY_EXAMPLES=64,PASSKEY_NUM_CHOICES=16,PASSKEY_GENERATOR_VERSION=synthetic_associative_passkey_v3,PASSKEY_VARIANT=confusable_records,PASSKEY_SCORE=normalized" \
  scripts/aggregate_kivi_passkey_axis.slurm)

printf 'powered_passkey_axis_eval_job=%s\n' "$eval_job"
printf 'powered_passkey_axis_input_aggregate_job=%s\n' "$input_aggregate"
printf 'powered_passkey_axis_comparison_job=%s\n' "$axis_aggregate"
printf 'powered_passkey_axis_output=%s\n' "$comparison_root"
