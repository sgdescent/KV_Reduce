#!/bin/bash
set -euo pipefail

dependency="${DEPENDENCY:-}"
dependency_args=()
if [[ -n "$dependency" ]]; then
  dependency_args+=(--dependency="afterok:${dependency}")
fi
out_root="${OUT_ROOT:-outputs/kivi_longbench_passage_retrieval_v1/qwen25_15b}"

array_job=$(sbatch --parsable \
  "${dependency_args[@]}" \
  --array=0-2%1 \
  --exclude=catalyst-0-9,catalyst-0-15 \
  --export="ALL,WANDB_PROJECT=${WANDB_PROJECT:-kv-reduce},LONGBENCH_EXAMPLES=${LONGBENCH_EXAMPLES:-24},OUT_ROOT=$out_root" \
  scripts/run_longbench_passage_retrieval.slurm)
aggregate_job=$(sbatch --parsable \
  --dependency="afterok:${array_job}" \
  --exclude=catalyst-0-9,catalyst-0-15 \
  --export="ALL,OUT_ROOT=$out_root" \
  scripts/aggregate_longbench_passage_retrieval.slurm)

printf 'longbench_retrieval_job=%s\n' "$array_job"
printf 'longbench_retrieval_aggregate_job=%s\n' "$aggregate_job"
printf 'longbench_retrieval_output=%s\n' "$out_root"
