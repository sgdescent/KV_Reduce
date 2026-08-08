#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs outputs/iclr_spec_kv

ARRAY_RANGE="${ARRAY_RANGE:-1-3}"
CONCURRENCY="${CONCURRENCY:-1}"
EXCLUDE_NODES="${EXCLUDE_NODES:-catalyst-0-9,catalyst-0-15}"
WANDB_PROJECT="${WANDB_PROJECT:-kv-reduce}"
RUN_SCRIPT="scripts/run_spec_kv_cross_family.slurm"
COMMON_EXPORT="ALL,ENABLE_WANDB=1,WANDB_PROJECT=${WANDB_PROJECT},BIG_DEVICE=cuda:0,SMALL_DEVICE=cuda:1"

smoke_job=$(sbatch --parsable \
  --partition=short \
  --time=01:00:00 \
  --gres=gpu:2 \
  --exclude="$EXCLUDE_NODES" \
  --array="${ARRAY_RANGE}%${CONCURRENCY}" \
  --export="${COMMON_EXPORT},STAGE=smoke" \
  "$RUN_SCRIPT")

main_job=$(sbatch --parsable \
  --partition=general \
  --time=12:00:00 \
  --gres=gpu:2 \
  --exclude="$EXCLUDE_NODES" \
  --array="${ARRAY_RANGE}%${CONCURRENCY}" \
  --dependency="afterany:${smoke_job}" \
  --kill-on-invalid-dep=yes \
  --export="${COMMON_EXPORT},STAGE=main" \
  "$RUN_SCRIPT")

sensitivity_job=$(sbatch --parsable \
  --partition=general \
  --time=12:00:00 \
  --gres=gpu:2 \
  --exclude="$EXCLUDE_NODES" \
  --array="${ARRAY_RANGE}%${CONCURRENCY}" \
  --dependency="afterany:${main_job}" \
  --kill-on-invalid-dep=yes \
  --export="${COMMON_EXPORT},STAGE=sensitivity" \
  "$RUN_SCRIPT")

allocation_job=$(sbatch --parsable \
  --partition=general \
  --time=08:00:00 \
  --gres=gpu:2 \
  --exclude="$EXCLUDE_NODES" \
  --array="${ARRAY_RANGE}%${CONCURRENCY}" \
  --dependency="afterany:${sensitivity_job}" \
  --kill-on-invalid-dep=yes \
  --export="${COMMON_EXPORT},STAGE=allocation" \
  "$RUN_SCRIPT")

robustness_job=$(sbatch --parsable \
  --partition=general \
  --time=08:00:00 \
  --gres=gpu:2 \
  --exclude="$EXCLUDE_NODES" \
  --array="3%1" \
  --dependency="afterany:${allocation_job}" \
  --kill-on-invalid-dep=yes \
  --export="${COMMON_EXPORT},STAGE=robustness" \
  "$RUN_SCRIPT")

printf 'two_gpu_smoke_job=%s\n' "$smoke_job"
printf 'two_gpu_main_job=%s\n' "$main_job"
printf 'two_gpu_sensitivity_job=%s\n' "$sensitivity_job"
printf 'two_gpu_allocation_job=%s\n' "$allocation_job"
printf 'two_gpu_robustness_job=%s\n' "$robustness_job"
printf 'monitor: squeue -j %s,%s,%s,%s,%s\n' \
  "$smoke_job" "$main_job" "$sensitivity_job" "$allocation_job" "$robustness_job"
