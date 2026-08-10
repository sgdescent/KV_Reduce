#!/bin/bash
set -euo pipefail

dependency_args=()
if [[ -n "${AFTER_JOB:-}" ]]; then
  dependency_args+=(--dependency="afterok:${AFTER_JOB}")
fi
exclude="${EXCLUDE_NODES:-catalyst-0-9,catalyst-0-15}"

one_gpu_spec=$(sbatch --parsable \
  "${dependency_args[@]}" \
  --array="0,4-6,10-11%1" \
  --exclude="$exclude" \
  --export=ALL,WANDB_PROJECT=kv-reduce \
  scripts/run_kivi_cross_family_spec.slurm)
two_gpu_spec=$(sbatch --parsable \
  "${dependency_args[@]}" \
  --gres=gpu:2 \
  --array="1-3,7-9%1" \
  --exclude="$exclude" \
  --export=ALL,BIG_DEVICE=cuda:0,SMALL_DEVICE=cuda:1,WANDB_PROJECT=kv-reduce \
  scripts/run_kivi_cross_family_spec.slurm)
quality=$(sbatch --parsable \
  "${dependency_args[@]}" \
  --array="0-11%2" \
  --exclude="$exclude" \
  --export=ALL,WANDB_PROJECT=kv-reduce \
  scripts/run_kivi_cross_family_quality.slurm)
aggregate=$(sbatch --parsable \
  --dependency="afterok:${one_gpu_spec}:${two_gpu_spec}:${quality}" \
  --array="0-5%2" \
  --exclude="$exclude" \
  scripts/aggregate_kivi_cross_family.slurm)

printf 'Submitted matched cross-family KIVI campaign\n  one-GPU spec: %s\n  two-GPU spec: %s\n  quality: %s\n  aggregate: %s\n' \
  "$one_gpu_spec" "$two_gpu_spec" "$quality" "$aggregate"
