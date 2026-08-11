#!/bin/bash
set -euo pipefail

cd "${REPO_ROOT:-/home/sakshamg/KV_Reduce}"
mkdir -p logs outputs/objective_kv

root="${OUT_ROOT:-outputs/objective_kv/qwen25_exact_objective_matrix_v1}"
exclude="${EXCLUDE_NODES:-catalyst-0-9,catalyst-0-15}"
big_model="${BIG_MODEL:-Qwen/Qwen2.5-3B}"
small_model="${SMALL_MODEL:-Qwen/Qwen2.5-1.5B}"
budgets="${BUDGETS:-6;8}"
contexts="${CONTEXTS:-1024;4096}"
seeds="${SEEDS:-0;1;2}"
num_profile="${NUM_PROFILE:-16}"
num_eval="${NUM_EVAL:-64}"
profile_continuation_len="${PROFILE_CONTINUATION_LEN:-16}"
profile_max_new_tokens="${PROFILE_MAX_NEW_TOKENS:-16}"
num_layers="${NUM_LAYERS:-28}"
layers="${LAYERS:-top:8}"
bits="${BITS:-8;4}"
dataset_name="${DATASET_NAME:-HuggingFaceFW/fineweb-edu}"
dataset_config="${DATASET_CONFIG:-sample-10BT}"
profile_skip="${PROFILE_SKIP:-60000}"
quality_eval_skip="${QUALITY_EVAL_SKIP:-70000}"
acceptance_eval_skip="${ACCEPTANCE_EVAL_SKIP:-72000}"
start_dependency="${START_DEPENDENCY:-}"
run_tag="${RUN_TAG:-qwen25_exact}"
profile_wandb_group="${PROFILE_WANDB_GROUP:-exact-objective-profile}"
matrix_wandb_group="${MATRIX_WANDB_GROUP:-exact-objective-matrix}"
matrix_throttle="${MATRIX_THROTTLE:-1}"

if ! [[ "$matrix_throttle" =~ ^[1-9][0-9]*$ ]]; then
  echo "MATRIX_THROTTLE must be a positive integer, got: $matrix_throttle" >&2
  exit 1
fi

if [[ "$profile_continuation_len" != "$profile_max_new_tokens" ]]; then
  echo "PROFILE_CONTINUATION_LEN and PROFILE_MAX_NEW_TOKENS must match for byte-aware allocation." >&2
  exit 1
fi

quality_profile="$root/quality_profile"
acceptance_profile="$root/acceptance_profile"
matrix_root="$root/matrix"
mkdir -p "$root"

dependency_args=()
if [[ -n "$start_dependency" ]]; then
  dependency_args+=(--dependency="afterok:$start_dependency")
fi

quality_job=$(sbatch --parsable --exclude="$exclude" "${dependency_args[@]}" \
  --export=ALL,MODEL="$small_model",DATASET_NAME="$dataset_name",DATASET_CONFIG="$dataset_config",EVAL_SPLIT=train,STREAM_EVAL=1,PROMPT_LEN=1024,CONTINUATION_LEN="$profile_continuation_len",NUM_SEQUENCES="$num_profile",SKIP_SEQUENCES="$profile_skip",LAYERS="$layers",BITS="$bits",QUALITY_RISK_METRIC=kl,KEY_QUANT_AXIS=per_channel,KEY_GROUP_SIZE=32,KEY_RESIDUAL_LENGTH=128,VALUE_QUANT_SCHEME=affine,OUT_DIR="$quality_profile",ENABLE_WANDB=1,WANDB_PROJECT=kv-reduce,WANDB_GROUP="$profile_wandb_group",WANDB_RUN_NAME="${run_tag}_quality_profile" \
  scripts/profile_kv_quality_sensitivity.slurm)

# Serialize every GPU stage so this campaign consumes at most one cluster GPU.
acceptance_job=$(sbatch --parsable --exclude="$exclude" --dependency="afterok:$quality_job" \
  --export=ALL,BIG_MODEL="$big_model",SMALL_MODEL="$small_model",DATASET_NAME="$dataset_name",DATASET_CONFIG="$dataset_config",EVAL_SPLIT=train,STREAM_EVAL=1,PROMPT_LEN=1024,NUM_PROMPTS="$num_profile",WARMUP_PROMPTS=0,SKIP_PROMPTS="$((profile_skip + num_profile + 128))",DRAFT_STEPS=4,MAX_NEW_TOKENS="$profile_max_new_tokens",LAYERS="$layers",BITS="$bits",TARGET_VERIFICATION_MODE=sequential,KEY_QUANT_AXIS=per_channel,KEY_GROUP_SIZE=32,KEY_RESIDUAL_LENGTH=128,VALUE_QUANT_SCHEME=affine,OUT_DIR="$acceptance_profile",ENABLE_WANDB=1,WANDB_PROJECT=kv-reduce,WANDB_GROUP="$profile_wandb_group",WANDB_RUN_NAME="${run_tag}_acceptance_profile" \
  scripts/profile_spec_kv_sensitivity.slurm)

prepare_job=$(sbatch --parsable --exclude="$exclude" --dependency="afterok:$acceptance_job" \
  --export=ALL,QUALITY_PROFILE_CSV="$quality_profile/profile_summary.csv",ACCEPTANCE_PROFILE_CSV="$acceptance_profile/profile_summary.csv",NUM_LAYERS="$num_layers",BUDGETS="$budgets",CONTEXTS="$contexts",SEEDS="$seeds",ALLOWED_BITS="$bits",NUM_EVAL="$num_eval",QUALITY_SKIP_BASE="$quality_eval_skip",ACCEPTANCE_SKIP_BASE="$acceptance_eval_skip",ACCEPTANCE_WARMUP_PROMPTS=0,QUALITY_RISK_FIELD=quality_risk,ACCEPTANCE_RISK_FIELD=accept_rate_drop_ucb95_clipped,OUT_DIR="$matrix_root" \
  scripts/prepare_objective_kv_matrix.slurm)

count_items() {
  local value="${1//;/,}"
  awk -F',' '{print NF}' <<< "$value"
}
num_tasks=$((2 * $(count_items "$budgets") * $(count_items "$contexts") * $(count_items "$seeds")))

matrix_job=$(sbatch --parsable --exclude="$exclude" --dependency="afterok:$prepare_job" \
  --array="0-$((num_tasks - 1))%$matrix_throttle" \
  --export=ALL,MANIFEST="$matrix_root/manifest.tsv",BIG_MODEL="$big_model",SMALL_MODEL="$small_model",DATASET_NAME="$dataset_name",DATASET_CONFIG="$dataset_config",EVAL_SPLIT=train,STREAM_EVAL=1,CONTINUATION_LEN=32,DRAFT_STEPS=4,MAX_NEW_TOKENS=16,TARGET_VERIFICATION_MODE=sequential,KEY_QUANT_AXIS=per_channel,KEY_GROUP_SIZE=32,KEY_RESIDUAL_LENGTH=128,VALUE_QUANT_SCHEME=affine,ENABLE_WANDB=1,WANDB_PROJECT=kv-reduce,WANDB_GROUP="$matrix_wandb_group" \
  scripts/eval_objective_kv_matrix.slurm)

aggregate_job=$(sbatch --parsable --exclude="$exclude" --dependency="afterok:$matrix_job" \
  --export=ALL,MATRIX_DIR="$matrix_root",OUT_DIR="$root/aggregate" \
  scripts/aggregate_objective_kv_matrix.slurm)

cat <<EOF
Submitted exact objective-aware KV matrix
  quality profile:    $quality_job
  acceptance profile: $acceptance_job
  prepare:            $prepare_job
  matrix ($num_tasks, max $matrix_throttle concurrent): $matrix_job
  aggregate:          $aggregate_job
  output:             $root
EOF
