#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs outputs/objective_kv

ALLOCATION_ROOT="${ALLOCATION_ROOT:?Set ALLOCATION_ROOT to a prepared objective matrix}"
MATRIX_ROOT="${MATRIX_ROOT:-outputs/objective_kv/gamma_matrix_v1}"
BUDGETS="${BUDGETS:-6,8}"
CONTEXTS="${CONTEXTS:-1024,4096}"
SEEDS="${SEEDS:-0,1,2}"
DRAFT_STEPS_LIST="${DRAFT_STEPS_LIST:-2,4,8}"
NUM_EVAL="${NUM_EVAL:-32}"
MAX_CONCURRENT="${MAX_CONCURRENT:-2}"
ENABLE_WANDB="${ENABLE_WANDB:-0}"
EXCLUDE_NODES="${EXCLUDE_NODES:-catalyst-0-9,catalyst-0-15}"
SOURCE_DEPENDENCY="${SOURCE_DEPENDENCY:-}"

count_items() {
  local normalized="${1//;/,}"
  awk -F, '{print NF}' <<< "$normalized"
}
num_tasks=$(( $(count_items "$BUDGETS") * $(count_items "$CONTEXTS") * $(count_items "$SEEDS") * $(count_items "$DRAFT_STEPS_LIST") ))

dependency_args=()
if [[ -n "$SOURCE_DEPENDENCY" ]]; then
  dependency_args+=(--dependency="afterok:$SOURCE_DEPENDENCY")
fi
prep=$(sbatch --parsable --exclude="$EXCLUDE_NODES" "${dependency_args[@]}" \
  --export=ALL,ALLOCATION_ROOT="$ALLOCATION_ROOT",BUDGETS="${BUDGETS//,/;}",CONTEXTS="${CONTEXTS//,/;}",SEEDS="${SEEDS//,/;}",DRAFT_STEPS_LIST="${DRAFT_STEPS_LIST//,/;}",NUM_EVAL="$NUM_EVAL",OUT_DIR="$MATRIX_ROOT" \
  scripts/prepare_spec_kv_gamma_matrix.slurm)
array=$(sbatch --parsable --exclude="$EXCLUDE_NODES" --dependency="afterok:$prep" \
  --array="0-$((num_tasks - 1))%$MAX_CONCURRENT" \
  --export=ALL,MANIFEST="$MATRIX_ROOT/manifest.tsv",ENABLE_WANDB="$ENABLE_WANDB",WANDB_PROJECT=kv-reduce,WANDB_GROUP=objective-gamma \
  scripts/eval_spec_kv_gamma_matrix.slurm)
aggregate=$(sbatch --parsable --exclude="$EXCLUDE_NODES" --dependency="afterok:$array" \
  --export=ALL,MATRIX_DIR="$MATRIX_ROOT",OUT_DIR="$MATRIX_ROOT/aggregate" \
  scripts/aggregate_spec_kv_gamma_matrix.slurm)

printf 'Prepared gamma matrix\n  prepare: %s\n  array: %s (%s tasks, max %s concurrent GPUs)\n  aggregate: %s\n  output: %s\n' \
  "$prep" "$array" "$num_tasks" "$MAX_CONCURRENT" "$aggregate" "$MATRIX_ROOT"
