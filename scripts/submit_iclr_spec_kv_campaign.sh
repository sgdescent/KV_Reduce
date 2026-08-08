#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs outputs/iclr_spec_kv

ARRAY_RANGE="${ARRAY_RANGE:-0-5}"
ROBUSTNESS_ARRAY_RANGE="${ROBUSTNESS_ARRAY_RANGE:-0,3,4}"
LONG_CONTEXT_ARRAY_RANGE="${LONG_CONTEXT_ARRAY_RANGE:-0}"
SMOKE_CONCURRENCY="${SMOKE_CONCURRENCY:-2}"
CAMPAIGN_CONCURRENCY="${CAMPAIGN_CONCURRENCY:-2}"
EXCLUDE_NODES="${EXCLUDE_NODES:-catalyst-0-9,catalyst-0-15}"
WANDB_PROJECT="${WANDB_PROJECT:-kv-reduce}"
RUN_SCRIPT="scripts/run_spec_kv_cross_family.slurm"

smoke_job=$(sbatch --parsable \
  --partition=short \
  --time=00:45:00 \
  --exclude="$EXCLUDE_NODES" \
  --array="${ARRAY_RANGE}%${SMOKE_CONCURRENCY}" \
  --export="ALL,STAGE=smoke,ENABLE_WANDB=1,WANDB_PROJECT=${WANDB_PROJECT}" \
  "$RUN_SCRIPT")

main_job=$(sbatch --parsable \
  --partition=general \
  --time=12:00:00 \
  --exclude="$EXCLUDE_NODES" \
  --array="${ARRAY_RANGE}%${CAMPAIGN_CONCURRENCY}" \
  --dependency="afterany:${smoke_job}" \
  --kill-on-invalid-dep=yes \
  --export="ALL,STAGE=main,ENABLE_WANDB=1,WANDB_PROJECT=${WANDB_PROJECT}" \
  "$RUN_SCRIPT")

sensitivity_job=$(sbatch --parsable \
  --partition=general \
  --time=12:00:00 \
  --exclude="$EXCLUDE_NODES" \
  --array="${ARRAY_RANGE}%${CAMPAIGN_CONCURRENCY}" \
  --dependency="afterany:${main_job}" \
  --kill-on-invalid-dep=yes \
  --export="ALL,STAGE=sensitivity,ENABLE_WANDB=1,WANDB_PROJECT=${WANDB_PROJECT}" \
  "$RUN_SCRIPT")

allocation_job=$(sbatch --parsable \
  --partition=general \
  --time=08:00:00 \
  --exclude="$EXCLUDE_NODES" \
  --array="${ARRAY_RANGE}%${CAMPAIGN_CONCURRENCY}" \
  --dependency="afterany:${sensitivity_job}" \
  --kill-on-invalid-dep=yes \
  --export="ALL,STAGE=allocation,ENABLE_WANDB=1,WANDB_PROJECT=${WANDB_PROJECT}" \
  "$RUN_SCRIPT")

robustness_job=$(sbatch --parsable \
  --partition=general \
  --time=08:00:00 \
  --exclude="$EXCLUDE_NODES" \
  --array="${ROBUSTNESS_ARRAY_RANGE}%${CAMPAIGN_CONCURRENCY}" \
  --dependency="afterany:${allocation_job}" \
  --kill-on-invalid-dep=yes \
  --export="ALL,STAGE=robustness,ENABLE_WANDB=1,WANDB_PROJECT=${WANDB_PROJECT}" \
  "$RUN_SCRIPT")

long_context_job=$(sbatch --parsable \
  --partition=general \
  --time=08:00:00 \
  --exclude="$EXCLUDE_NODES" \
  --array="${LONG_CONTEXT_ARRAY_RANGE}%1" \
  --dependency="afterany:${robustness_job}" \
  --kill-on-invalid-dep=yes \
  --export="ALL,STAGE=long_context,ENABLE_WANDB=1,WANDB_PROJECT=${WANDB_PROJECT}" \
  "$RUN_SCRIPT")

printf 'smoke_job=%s\n' "$smoke_job"
printf 'main_job=%s\n' "$main_job"
printf 'sensitivity_job=%s\n' "$sensitivity_job"
printf 'allocation_job=%s\n' "$allocation_job"
printf 'robustness_job=%s\n' "$robustness_job"
printf 'long_context_job=%s\n' "$long_context_job"
printf 'monitor: squeue -j %s,%s,%s,%s,%s,%s\n' \
  "$smoke_job" "$main_job" "$sensitivity_job" "$allocation_job" "$robustness_job" "$long_context_job"
