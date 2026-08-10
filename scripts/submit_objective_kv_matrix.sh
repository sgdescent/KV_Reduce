#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs outputs/objective_kv

SOURCE_ROOT="${SOURCE_ROOT:?Set SOURCE_ROOT to a completed objective profiling campaign}"
MATRIX_ROOT="${MATRIX_ROOT:-outputs/objective_kv/matrix_v1}"
BUDGETS="${BUDGETS:-6,8,10,12}"
CONTEXTS="${CONTEXTS:-512,1024,4096}"
SEEDS="${SEEDS:-0,1,2}"
NUM_EVAL="${NUM_EVAL:-32}"
QUALITY_SKIP_BASE="${QUALITY_SKIP_BASE:-0}"
ACCEPTANCE_SKIP_BASE="${ACCEPTANCE_SKIP_BASE:-0}"
ACCEPTANCE_WARMUP_PROMPTS="${ACCEPTANCE_WARMUP_PROMPTS:-2}"
NUM_LAYERS="${NUM_LAYERS:-28}"
MAX_CONCURRENT="${MAX_CONCURRENT:-2}"
ENABLE_WANDB="${ENABLE_WANDB:-1}"
WANDB_GROUP="${WANDB_GROUP:-objective-matrix}"
EXCLUDE_NODES="${EXCLUDE_NODES:-catalyst-0-9,catalyst-0-15}"
SOURCE_DEPENDENCY="${SOURCE_DEPENDENCY:-}"
QUALITY_PROFILE_CSV="${QUALITY_PROFILE_CSV:-$SOURCE_ROOT/quality_profile/profile_summary.csv}"
ACCEPTANCE_PROFILE_CSV="${ACCEPTANCE_PROFILE_CSV:-$SOURCE_ROOT/acceptance_profile/profile_summary.csv}"
QUALITY_RISK_FIELD="${QUALITY_RISK_FIELD:-quality_risk}"
ACCEPTANCE_RISK_FIELD="${ACCEPTANCE_RISK_FIELD:-accept_rate_drop}"

count_items() {
  local normalized="${1//;/,}"
  awk -F, '{print NF}' <<< "$normalized"
}
num_tasks=$(( $(count_items "$BUDGETS") * $(count_items "$CONTEXTS") * $(count_items "$SEEDS") * 2 ))

dependency_args=()
if [[ -n "$SOURCE_DEPENDENCY" ]]; then
  dependency_args+=(--dependency="afterok:$SOURCE_DEPENDENCY")
fi
prep=$(sbatch --parsable --exclude="$EXCLUDE_NODES" "${dependency_args[@]}" \
  --export=ALL,QUALITY_PROFILE_CSV="$QUALITY_PROFILE_CSV",ACCEPTANCE_PROFILE_CSV="$ACCEPTANCE_PROFILE_CSV",QUALITY_RISK_FIELD="$QUALITY_RISK_FIELD",ACCEPTANCE_RISK_FIELD="$ACCEPTANCE_RISK_FIELD",NUM_LAYERS="$NUM_LAYERS",BUDGETS="${BUDGETS//,/;}",CONTEXTS="${CONTEXTS//,/;}",SEEDS="${SEEDS//,/;}",NUM_EVAL="$NUM_EVAL",QUALITY_SKIP_BASE="$QUALITY_SKIP_BASE",ACCEPTANCE_SKIP_BASE="$ACCEPTANCE_SKIP_BASE",ACCEPTANCE_WARMUP_PROMPTS="$ACCEPTANCE_WARMUP_PROMPTS",OUT_DIR="$MATRIX_ROOT" \
  scripts/prepare_objective_kv_matrix.slurm)
array=$(sbatch --parsable --exclude="$EXCLUDE_NODES" --dependency="afterok:$prep" \
  --array="0-$((num_tasks - 1))%$MAX_CONCURRENT" \
  --export=ALL,MANIFEST="$MATRIX_ROOT/manifest.tsv",ENABLE_WANDB="$ENABLE_WANDB",WANDB_PROJECT=kv-reduce,WANDB_GROUP="$WANDB_GROUP" \
  scripts/eval_objective_kv_matrix.slurm)
aggregate=$(sbatch --parsable --exclude="$EXCLUDE_NODES" --dependency="afterok:$array" \
  --export=ALL,MATRIX_DIR="$MATRIX_ROOT",OUT_DIR="$MATRIX_ROOT/aggregate" \
  scripts/aggregate_objective_kv_matrix.slurm)

printf 'Prepared objective matrix campaign\n  prepare: %s\n  array: %s (%s tasks, max %s concurrent GPUs)\n  aggregate: %s\n  output: %s\n' \
  "$prep" "$array" "$num_tasks" "$MAX_CONCURRENT" "$aggregate" "$MATRIX_ROOT"
