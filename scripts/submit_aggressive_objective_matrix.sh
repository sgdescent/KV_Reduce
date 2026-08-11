#!/bin/bash
set -euo pipefail

# Exact low-bit objective cross-evaluation. START_DEPENDENCY can serialize this
# behind another campaign without consuming an additional live GPU slot.
export OUT_ROOT="${OUT_ROOT:-outputs/objective_kv/qwen25_exact_aggressive_objective_matrix_v1}"
export BUDGETS="${BUDGETS:-3;5}"
export BITS="${BITS:-8;4;2}"
export CONTEXTS="${CONTEXTS:-1024;4096}"
export SEEDS="${SEEDS:-0;1;2}"
export NUM_PROFILE="${NUM_PROFILE:-32}"
export NUM_EVAL="${NUM_EVAL:-64}"
export RUN_TAG="${RUN_TAG:-qwen25_exact_aggressive}"
export PROFILE_WANDB_GROUP="${PROFILE_WANDB_GROUP:-exact-aggressive-objective-profile}"
export MATRIX_WANDB_GROUP="${MATRIX_WANDB_GROUP:-exact-aggressive-objective-matrix}"

exec bash scripts/submit_exact_objective_matrix.sh
