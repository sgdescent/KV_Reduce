#!/usr/bin/env bash
# Submit a clean O/value-reader budget sweep for absorbed KV Reduce.
#
# Example:
#   bash scripts/submit_absorbed_o_sweep.sh
#
# Useful overrides:
#   O_BUDGETS="2048 8192 20000" SHARED_SPECS="top:4 top:8" ENABLE_WANDB=1 bash scripts/submit_absorbed_o_sweep.sh

set -euo pipefail

EXCLUDE_NODES="${EXCLUDE_NODES:-catalyst-0-9}"
RUN_TAG="${RUN_TAG:-o_sweep_$(date +%Y%m%d_%H%M%S)}"

BIG_MODEL="${BIG_MODEL:-Qwen/Qwen2.5-3B}"
SMALL_MODEL="${SMALL_MODEL:-Qwen/Qwen2.5-1.5B}"
DATASET_NAME="${DATASET_NAME:-HuggingFaceFW/fineweb-edu}"
DATASET_CONFIG="${DATASET_CONFIG:-sample-10BT}"
SEQ_LEN="${SEQ_LEN:-512}"
K_TRAIN_SEQUENCES="${K_TRAIN_SEQUENCES:-256}"
O_BUDGETS="${O_BUDGETS:-2048 8192 20000}"
LAMBDA_REG="${LAMBDA_REG:-1e-4}"
OUTPUT_ROUTING_SOURCE="${OUTPUT_ROUTING_SOURCE:-shared}"
LAYER_MAP_FILE="${LAYER_MAP_FILE:-outputs/layer_map_qwen25_3b_15b_revamp/layer_map.json}"

BENCH_DATASET_NAME="${BENCH_DATASET_NAME:-wikitext}"
BENCH_DATASET_CONFIG="${BENCH_DATASET_CONFIG:-wikitext-2-raw-v1}"
PROMPT_LEN="${PROMPT_LEN:-1024}"
NUM_PROMPTS="${NUM_PROMPTS:-200}"
WARMUP_PROMPTS="${WARMUP_PROMPTS:-10}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
DRAFT_STEPS="${DRAFT_STEPS:-4}"
SHARED_SPECS="${SHARED_SPECS:-top:4 top:8}"
NORM_MATCH="${NORM_MATCH:-rms}"
ABSORBED_CACHE_MODE="${ABSORBED_CACHE_MODE:-prefix}"

ENABLE_WANDB="${ENABLE_WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-kv-reduce}"
WANDB_GROUP="${WANDB_GROUP:-absorbed-o-sweep-${RUN_TAG}}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

mkdir -p logs outputs

echo "Submitting absorbed O/value-reader sweep"
echo "  RUN_TAG=$RUN_TAG"
echo "  O_BUDGETS=$O_BUDGETS"
echo "  SHARED_SPECS=$SHARED_SPECS"
echo "  WANDB_GROUP=$WANDB_GROUP"

for o_budget in $O_BUDGETS; do
  fit_out="outputs/kv_absorbed_${RUN_TAG}_k${K_TRAIN_SEQUENCES}_o${o_budget}"
  fit_name="fit_k${K_TRAIN_SEQUENCES}_o${o_budget}"
  fit_job=$(
    sbatch --parsable \
      --exclude="$EXCLUDE_NODES" \
      --job-name="$fit_name" \
      --export=ALL,ENABLE_WANDB="$ENABLE_WANDB",WANDB_PROJECT="$WANDB_PROJECT",WANDB_GROUP="$WANDB_GROUP",WANDB_ENTITY="$WANDB_ENTITY",WANDB_RUN_NAME="$fit_name",BIG_MODEL="$BIG_MODEL",SMALL_MODEL="$SMALL_MODEL",DATASET_NAME="$DATASET_NAME",DATASET_CONFIG="$DATASET_CONFIG",SEQ_LEN="$SEQ_LEN",TRAIN_SEQUENCES="$o_budget",K_TRAIN_SEQUENCES="$K_TRAIN_SEQUENCES",O_TRAIN_SEQUENCES="$o_budget",LAMBDA_REG="$LAMBDA_REG",OUTPUT_ROUTING_SOURCE="$OUTPUT_ROUTING_SOURCE",LAYER_MAP_FILE="$LAYER_MAP_FILE",OUT_DIR="$fit_out" \
      scripts/fit_kv_absorbed_learned_map.slurm
  )
  echo "  fit O=$o_budget -> job $fit_job -> $fit_out"

  for shared_spec in $SHARED_SPECS; do
    safe_spec="${shared_spec//:/}"
    bench_out="outputs/benchmark_${RUN_TAG}_k${K_TRAIN_SEQUENCES}_o${o_budget}_${safe_spec}"
    bench_name="bench_o${o_budget}_${safe_spec}"
    bench_job=$(
      sbatch --parsable \
        --dependency=afterok:"$fit_job" \
        --exclude="$EXCLUDE_NODES" \
        --job-name="$bench_name" \
        --export=ALL,ENABLE_WANDB="$ENABLE_WANDB",WANDB_PROJECT="$WANDB_PROJECT",WANDB_GROUP="$WANDB_GROUP",WANDB_ENTITY="$WANDB_ENTITY",WANDB_RUN_NAME="$bench_name",BIG_MODEL="$BIG_MODEL",SMALL_MODEL="$SMALL_MODEL",TRANSLATOR_PATH="$fit_out/absorbed_translator.pt",DATASET_NAME="$BENCH_DATASET_NAME",DATASET_CONFIG="$BENCH_DATASET_CONFIG",PROMPT_LEN="$PROMPT_LEN",NUM_PROMPTS="$NUM_PROMPTS",WARMUP_PROMPTS="$WARMUP_PROMPTS",MAX_NEW_TOKENS="$MAX_NEW_TOKENS",DRAFT_STEPS="$DRAFT_STEPS",SHARED_LAYERS="$shared_spec",NORM_MATCH="$NORM_MATCH",ABSORBED_CACHE_MODE="$ABSORBED_CACHE_MODE",OUT_DIR="$bench_out" \
        scripts/benchmark_absorbed_cached.slurm
    )
    echo "    benchmark $shared_spec -> job $bench_job -> $bench_out"
  done
done
