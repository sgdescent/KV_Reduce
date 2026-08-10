#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs outputs/value_precision_sweep

exclude_nodes="${EXCLUDE_NODES:-catalyst-0-9,catalyst-0-15}"
sweep_root="${SWEEP_ROOT:-outputs/value_precision_sweep/qwen25_3b_15b}"
array_job=$(sbatch --parsable \
  --exclude="$exclude_nodes" \
  --array="0-8%${MAX_CONCURRENT:-1}" \
  --export=ALL,SWEEP_ROOT="$sweep_root",WANDB_PROJECT="${WANDB_PROJECT:-kv-reduce}",WANDB_GROUP="${WANDB_GROUP:-value-precision-sweep}" \
  scripts/run_value_precision_sweep.slurm)
aggregate_job=$(sbatch --parsable \
  --exclude="$exclude_nodes" \
  --dependency="afterok:$array_job" \
  --export=ALL,SWEEP_ROOT="$sweep_root",OUT_DIR="$sweep_root/aggregate" \
  scripts/aggregate_value_precision_sweep.slurm)

printf 'Value-precision sweep submitted\n  array: %s\n  aggregate: %s\n  output: %s\n' \
  "$array_job" "$aggregate_job" "$sweep_root"
