#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs outputs/objective_kv

TAG="${TAG:-qwen25_3b_15b_1k}"
ROOT="${ROOT:-outputs/objective_kv/$TAG}"
BIG_MODEL="${BIG_MODEL:-Qwen/Qwen2.5-3B}"
SMALL_MODEL="${SMALL_MODEL:-Qwen/Qwen2.5-1.5B}"
PROMPT_LEN="${PROMPT_LEN:-1024}"
CONTINUATION_LEN="${CONTINUATION_LEN:-32}"
NUM_PROFILE="${NUM_PROFILE:-16}"
NUM_EVAL="${NUM_EVAL:-32}"
NUM_LAYERS="${NUM_LAYERS:-28}"
LAYERS="${LAYERS:-top:8}"
BITS="${BITS:-8,4}"
EXPORT_BITS="${BITS//,/;}"
# Semicolons survive Slurm's comma-delimited --export syntax and are accepted
# by the shared bit-list parser.
ALLOWED_BITS="${ALLOWED_BITS:-$EXPORT_BITS;16}"
TARGET_PROFILED_MEAN_BITS="${TARGET_PROFILED_MEAN_BITS:-8}"
EXCLUDE_NODES="${EXCLUDE_NODES:-catalyst-0-9,catalyst-0-15}"
WANDB_PROJECT="${WANDB_PROJECT:-kv-reduce}"
ENABLE_WANDB="${ENABLE_WANDB:-1}"
START_DEPENDENCY="${START_DEPENDENCY:-}"
ACCEPTANCE_RISK_FIELD="${ACCEPTANCE_RISK_FIELD:-accept_rate_drop}"
KEY_QUANT_AXIS="${KEY_QUANT_AXIS:-per_channel}"
KEY_GROUP_SIZE="${KEY_GROUP_SIZE:-32}"
KEY_RESIDUAL_LENGTH="${KEY_RESIDUAL_LENGTH:-128}"
VALUE_QUANT_SCHEME="${VALUE_QUANT_SCHEME:-affine}"
PROFILE_SKIP="${PROFILE_SKIP:-0}"
QUALITY_EVAL_SKIP="${QUALITY_EVAL_SKIP:-$((PROFILE_SKIP + NUM_PROFILE + 1))}"
ACCEPTANCE_EVAL_SKIP="${ACCEPTANCE_EVAL_SKIP:-$((QUALITY_EVAL_SKIP + NUM_EVAL))}"

QUALITY_PROFILE="$ROOT/quality_profile"
ACCEPTANCE_PROFILE="$ROOT/acceptance_profile"
QUALITY_ALLOCATION="$ROOT/quality_allocation"
ACCEPTANCE_ALLOCATION="$ROOT/acceptance_allocation"
COMPARISON="$ROOT/objective_comparison"
QUALITY_EVAL="$ROOT/quality_cross_eval"
ACCEPTANCE_EVAL="$ROOT/acceptance_cross_eval"
FINAL_RESULTS="$ROOT/final_results"
mkdir -p "$ROOT"

submit() {
  sbatch --parsable --exclude="$EXCLUDE_NODES" "$@"
}

start_dependency_args=()
if [[ -n "$START_DEPENDENCY" ]]; then
  start_dependency_args+=(--dependency="afterok:$START_DEPENDENCY")
fi
quality_job=$(submit "${start_dependency_args[@]}" --export=ALL,MODEL="$SMALL_MODEL",PROMPT_LEN="$PROMPT_LEN",CONTINUATION_LEN="$CONTINUATION_LEN",NUM_SEQUENCES="$NUM_PROFILE",SKIP_SEQUENCES="$PROFILE_SKIP",LAYERS="$LAYERS",BITS="$EXPORT_BITS",KEY_QUANT_AXIS="$KEY_QUANT_AXIS",KEY_GROUP_SIZE="$KEY_GROUP_SIZE",KEY_RESIDUAL_LENGTH="$KEY_RESIDUAL_LENGTH",VALUE_QUANT_SCHEME="$VALUE_QUANT_SCHEME",OUT_DIR="$QUALITY_PROFILE",ENABLE_WANDB="$ENABLE_WANDB",WANDB_PROJECT="$WANDB_PROJECT",WANDB_GROUP=objective-profile,WANDB_RUN_NAME="${TAG}_quality_profile" scripts/profile_kv_quality_sensitivity.slurm)
acceptance_job=$(submit "${start_dependency_args[@]}" --export=ALL,BIG_MODEL="$BIG_MODEL",SMALL_MODEL="$SMALL_MODEL",PROMPT_LEN="$PROMPT_LEN",NUM_PROMPTS="$NUM_PROFILE",WARMUP_PROMPTS=1,SKIP_PROMPTS="$PROFILE_SKIP",LAYERS="$LAYERS",BITS="$EXPORT_BITS",KEY_QUANT_AXIS="$KEY_QUANT_AXIS",KEY_GROUP_SIZE="$KEY_GROUP_SIZE",KEY_RESIDUAL_LENGTH="$KEY_RESIDUAL_LENGTH",VALUE_QUANT_SCHEME="$VALUE_QUANT_SCHEME",OUT_DIR="$ACCEPTANCE_PROFILE",ENABLE_WANDB="$ENABLE_WANDB",WANDB_PROJECT="$WANDB_PROJECT",WANDB_GROUP=objective-profile,WANDB_RUN_NAME="${TAG}_acceptance_profile" scripts/profile_spec_kv_sensitivity.slurm)

quality_alloc_job=$(submit --dependency="afterok:$quality_job" --export=ALL,PROFILE_CSV="$QUALITY_PROFILE/profile_summary.csv",NUM_LAYERS="$NUM_LAYERS",ALLOWED_BITS="$ALLOWED_BITS",RISK_FIELD=quality_risk,TARGET_PROFILED_MEAN_BITS="$TARGET_PROFILED_MEAN_BITS",NAME=quality_optimized,OUT_DIR="$QUALITY_ALLOCATION" scripts/search_kv_bit_allocation.slurm)
acceptance_alloc_job=$(submit --dependency="afterok:$acceptance_job" --export=ALL,PROFILE_CSV="$ACCEPTANCE_PROFILE/profile_summary.csv",NUM_LAYERS="$NUM_LAYERS",ALLOWED_BITS="$ALLOWED_BITS",RISK_FIELD="$ACCEPTANCE_RISK_FIELD",TARGET_PROFILED_MEAN_BITS="$TARGET_PROFILED_MEAN_BITS",NAME=acceptance_optimized,OUT_DIR="$ACCEPTANCE_ALLOCATION" scripts/search_kv_bit_allocation.slurm)

both_allocations="afterok:$quality_alloc_job:$acceptance_alloc_job"
comparison_job=$(submit --dependency="$both_allocations" --export=ALL,QUALITY_PROFILE_CSV="$QUALITY_PROFILE/profile_summary.csv",ACCEPTANCE_PROFILE_CSV="$ACCEPTANCE_PROFILE/profile_summary.csv",QUALITY_ALLOCATION="$QUALITY_ALLOCATION/allocation.json",ACCEPTANCE_ALLOCATION="$ACCEPTANCE_ALLOCATION/allocation.json",OUT_DIR="$COMPARISON" scripts/compare_kv_objectives.slurm)

configs="none;allocation:$QUALITY_ALLOCATION/allocation.json;allocation:$ACCEPTANCE_ALLOCATION/allocation.json"
quality_eval_job=$(submit --dependency="$both_allocations" --export=ALL,MODEL="$SMALL_MODEL",PROMPT_LEN="$PROMPT_LEN",CONTINUATION_LEN="$CONTINUATION_LEN",NUM_SEQUENCES="$NUM_EVAL",SKIP_SEQUENCES="$QUALITY_EVAL_SKIP",QUANT_CONFIGS="$configs",KEY_QUANT_AXIS="$KEY_QUANT_AXIS",KEY_GROUP_SIZE="$KEY_GROUP_SIZE",KEY_RESIDUAL_LENGTH="$KEY_RESIDUAL_LENGTH",VALUE_QUANT_SCHEME="$VALUE_QUANT_SCHEME",OUT_DIR="$QUALITY_EVAL",ENABLE_WANDB="$ENABLE_WANDB",WANDB_PROJECT="$WANDB_PROJECT",WANDB_GROUP=objective-cross-eval,WANDB_RUN_NAME="${TAG}_quality_cross_eval" scripts/profile_kv_quality_sensitivity.slurm)
acceptance_eval_job=$(submit --dependency="$both_allocations" --export=ALL,BIG_MODEL="$BIG_MODEL",SMALL_MODEL="$SMALL_MODEL",PROMPT_LEN="$PROMPT_LEN",NUM_PROMPTS="$NUM_EVAL",WARMUP_PROMPTS=2,SKIP_PROMPTS="$ACCEPTANCE_EVAL_SKIP",QUANT_CONFIGS="$configs",KEY_QUANT_AXIS="$KEY_QUANT_AXIS",KEY_GROUP_SIZE="$KEY_GROUP_SIZE",KEY_RESIDUAL_LENGTH="$KEY_RESIDUAL_LENGTH",VALUE_QUANT_SCHEME="$VALUE_QUANT_SCHEME",OUT_DIR="$ACCEPTANCE_EVAL",ENABLE_WANDB="$ENABLE_WANDB",WANDB_PROJECT="$WANDB_PROJECT",WANDB_GROUP=objective-cross-eval,WANDB_RUN_NAME="${TAG}_acceptance_cross_eval" scripts/benchmark_spec_kv_quantization.slurm)
final_job=$(submit --dependency="afterok:$comparison_job:$quality_eval_job:$acceptance_eval_job" --export=ALL,QUALITY_SUMMARY="$QUALITY_EVAL/summary.json",ACCEPTANCE_SUMMARY="$ACCEPTANCE_EVAL/summary.json",QUALITY_ALLOCATION="$QUALITY_ALLOCATION/allocation.json",ACCEPTANCE_ALLOCATION="$ACCEPTANCE_ALLOCATION/allocation.json",COMPARISON_SUMMARY="$COMPARISON/summary.json",OUT_DIR="$FINAL_RESULTS" scripts/aggregate_objective_kv_results.slurm)

cat <<EOF
Submitted objective-aware KV campaign: $TAG
  quality profile:      $quality_job
  acceptance profile:   $acceptance_job
  quality allocation:   $quality_alloc_job
  acceptance allocation:$acceptance_alloc_job
  objective comparison: $comparison_job
  quality cross-eval:   $quality_eval_job
  acceptance cross-eval:$acceptance_eval_job
  final aggregation:    $final_job
  outputs:               $ROOT
EOF
