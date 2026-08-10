#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs outputs/value_precision_cross_family

EXCLUDE_NODES="${EXCLUDE_NODES:-catalyst-0-9,catalyst-0-15}"
WANDB_PROJECT="${WANDB_PROJECT:-kv-reduce}"
VALUE_SWEEP_ROOT="${VALUE_SWEEP_ROOT:-outputs/value_precision_cross_family}"
AFTER_JOBS="${AFTER_JOBS:-}"
dependency_args=()
if [[ -n "$AFTER_JOBS" ]]; then
  dependency_args+=(--dependency="afterany:${AFTER_JOBS}")
fi

# Qwen2.5-3B/1.5B, OLMo, and SmolLM fit comfortably on one GPU.
one_gpu_job=$(sbatch --parsable \
  "${dependency_args[@]}" \
  --array="0,4-6,10-12,16-17%1" \
  --exclude="$EXCLUDE_NODES" \
  --export="ALL,STAGE=value_precision,ENABLE_WANDB=1,WANDB_PROJECT=${WANDB_PROJECT},VALUE_SWEEP_ROOT=${VALUE_SWEEP_ROOT}" \
  scripts/run_spec_kv_cross_family.slurm)

# The larger target--draft pairs use separate devices and run one pair at a time.
two_gpu_job=$(sbatch --parsable \
  "${dependency_args[@]}" \
  --gres=gpu:2 \
  --array="1-3,7-9,13-15%1" \
  --exclude="$EXCLUDE_NODES" \
  --export="ALL,STAGE=value_precision,ENABLE_WANDB=1,WANDB_PROJECT=${WANDB_PROJECT},VALUE_SWEEP_ROOT=${VALUE_SWEEP_ROOT},BIG_DEVICE=cuda:0,SMALL_DEVICE=cuda:1" \
  scripts/run_spec_kv_cross_family.slurm)

aggregate_job=$(sbatch --parsable \
  --dependency="afterok:${one_gpu_job}:${two_gpu_job}" \
  --array="0-5%2" \
  --exclude="$EXCLUDE_NODES" \
  --export="ALL,VALUE_SWEEP_ROOT=${VALUE_SWEEP_ROOT}" \
  scripts/aggregate_value_precision_cross_family.slurm)

printf 'Submitted cross-family V3 confirmation\n  one GPU: %s\n  two GPU: %s\n  aggregate: %s\n' \
  "$one_gpu_job" "$two_gpu_job" "$aggregate_job"
