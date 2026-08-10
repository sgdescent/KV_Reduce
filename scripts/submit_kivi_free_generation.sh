#!/bin/bash
set -euo pipefail

exclude="${EXCLUDE_NODES:-catalyst-0-9,catalyst-0-15}"
root="${ROOT:-outputs/kivi_free_generation_v1}"
after_job="${AFTER_JOB:-}"
dependency=()
if [[ -n "$after_job" ]]; then
  dependency+=(--dependency="afterok:${after_job}")
fi

tags=(qwen25_15b llama32_3b olmo2_1b smollm2_360m)
models=(
  Qwen/Qwen2.5-1.5B
  meta-llama/Llama-3.2-3B
  allenai/OLMo-2-0425-1B
  HuggingFaceTB/SmolLM2-360M
)
jobs=()
for index in "${!tags[@]}"; do
  tag="${tags[$index]}"
  model="${models[$index]}"
  job=$(sbatch --parsable \
    "${dependency[@]}" \
    --array=0-2%1 \
    --exclude="$exclude" \
    --export="ALL,MODEL=${model},MODEL_TAG=${tag},OUT_ROOT=${root}/cross_family,NUM_PROMPTS=32,PROMPT_LEN=1024,MAX_NEW_TOKENS=64,SKIP_STRIDE=40,WANDB_PROJECT=kv-reduce,WANDB_GROUP=kivi-free-generation-cross-family" \
    scripts/run_kivi_free_generation.slurm)
  jobs+=("$job")
done

joined=$(IFS=:; echo "${jobs[*]}")
cross_agg=$(sbatch --parsable \
  --dependency="afterok:${joined}" \
  --exclude="$exclude" \
  --export="ALL,SUMMARY_GLOB=${root}/cross_family/**/seed_*/summary.json,OUT_DIR=${root}/cross_family_aggregate" \
  scripts/aggregate_kivi_free_generation.slurm)

long_gen=$(sbatch --parsable \
  --dependency="afterok:${cross_agg}" \
  --array=0-2%1 \
  --exclude="$exclude" \
  --export="ALL,MODEL=Qwen/Qwen2.5-1.5B,MODEL_TAG=qwen25_15b,OUT_ROOT=${root}/long_generation,NUM_PROMPTS=16,PROMPT_LEN=1024,MAX_NEW_TOKENS=256,SKIP_STRIDE=24,WANDB_PROJECT=kv-reduce,WANDB_GROUP=kivi-free-generation-length" \
  scripts/run_kivi_free_generation.slurm)

ctx4k=$(sbatch --parsable \
  --dependency="afterok:${cross_agg}" \
  --array=0-2%1 \
  --exclude="$exclude" \
  --export="ALL,MODEL=Qwen/Qwen2.5-1.5B,MODEL_TAG=qwen25_15b,OUT_ROOT=${root}/long_context,NUM_PROMPTS=16,PROMPT_LEN=4096,MAX_NEW_TOKENS=64,SKIP_STRIDE=24,WANDB_PROJECT=kv-reduce,WANDB_GROUP=kivi-free-generation-context" \
  scripts/run_kivi_free_generation.slurm)

ctx16k=$(sbatch --parsable \
  --dependency="afterok:${cross_agg}" \
  --array=0-2%1 \
  --exclude="$exclude" \
  --export="ALL,MODEL=Qwen/Qwen2.5-1.5B,MODEL_TAG=qwen25_15b,OUT_ROOT=${root}/long_context,NUM_PROMPTS=12,PROMPT_LEN=16384,MAX_NEW_TOKENS=64,SKIP_STRIDE=18,WANDB_PROJECT=kv-reduce,WANDB_GROUP=kivi-free-generation-context" \
  scripts/run_kivi_free_generation.slurm)

tail_joined="${long_gen}:${ctx4k}:${ctx16k}"
final_agg=$(sbatch --parsable \
  --dependency="afterok:${tail_joined}" \
  --exclude="$exclude" \
  --export="ALL,SUMMARY_GLOB=${root}/cross_family/**/seed_*/summary.json;${root}/long_generation/**/seed_*/summary.json;${root}/long_context/**/seed_*/summary.json,OUT_DIR=${root}/aggregate" \
  scripts/aggregate_kivi_free_generation.slurm)

printf 'Submitted free-running target-cache campaign\n'
printf '  cross-family arrays: %s\n' "${jobs[*]}"
printf '  cross-family aggregate: %s\n' "$cross_agg"
printf '  long-generation: %s\n' "$long_gen"
printf '  4K context: %s\n' "$ctx4k"
printf '  16K context: %s\n' "$ctx16k"
printf '  final aggregate: %s\n' "$final_agg"
