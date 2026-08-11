#!/bin/bash
set -euo pipefail

dependency="${DEPENDENCY:-}"
dependency_args=()
if [[ -n "$dependency" ]]; then
  dependency_args+=(--dependency="afterok:${dependency}")
fi
model="${MODEL:-Qwen/Qwen2.5-1.5B}"
model_label="${MODEL_LABEL:-qwen25_15b}"
max_prompt_tokens="${MAX_PROMPT_TOKENS:-32752}"
quant_configs="${QUANT_CONFIGS:-none;k8v4;k4v8;k4v4;k4v2;k2v4;k2v2}"
out_root="${OUT_ROOT:-outputs/kivi_longbench_passage_retrieval_v1/$model_label}"

array_job=$(sbatch --parsable \
  "${dependency_args[@]}" \
  --array=0-2%1 \
  --exclude=catalyst-0-9,catalyst-0-15 \
  --export="ALL,WANDB_PROJECT=${WANDB_PROJECT:-kv-reduce},LONGBENCH_EXAMPLES=${LONGBENCH_EXAMPLES:-24},MODEL=$model,MODEL_LABEL=$model_label,MAX_PROMPT_TOKENS=$max_prompt_tokens,QUANT_CONFIGS=$quant_configs,OUT_ROOT=$out_root" \
  scripts/run_longbench_passage_retrieval.slurm)
aggregate_job=$(sbatch --parsable \
  --dependency="afterok:${array_job}" \
  --exclude=catalyst-0-9,catalyst-0-15 \
  --export="ALL,OUT_ROOT=$out_root" \
  scripts/aggregate_longbench_passage_retrieval.slurm)

printf 'longbench_retrieval_job=%s\n' "$array_job"
printf 'longbench_retrieval_aggregate_job=%s\n' "$aggregate_job"
printf 'longbench_retrieval_output=%s\n' "$out_root"
