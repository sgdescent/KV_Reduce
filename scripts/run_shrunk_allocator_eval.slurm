#!/bin/bash
#SBATCH --job-name=kv_shrink_eval
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "${REPO_ROOT:-/home/sakshamg/KV_Reduce}"
source .venv/bin/activate

root="${OUT_ROOT:?Set OUT_ROOT}"
configs=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["quant_configs"])' "$root/bundle.json")
quality_out="$root/quality_eval"
acceptance_out="$root/acceptance_eval"
mkdir -p "$quality_out" "$acceptance_out"

common_quant=(
  --key_quant_axis "${KEY_QUANT_AXIS:-per_channel}"
  --key_group_size "${KEY_GROUP_SIZE:-32}"
  --key_residual_length "${KEY_RESIDUAL_LENGTH:-128}"
  --value_quant_scheme "${VALUE_QUANT_SCHEME:-affine}"
)

python3 -u profile_kv_quality_sensitivity.py \
  --model "${SMALL_MODEL:-Qwen/Qwen2.5-1.5B}" \
  --device cuda:0 \
  --dtype bf16 \
  --attn_implementation sdpa \
  --dataset_name "${DATASET_NAME:-HuggingFaceFW/fineweb-edu}" \
  --dataset_config "${DATASET_CONFIG:-sample-10BT}" \
  --eval_split "${EVAL_SPLIT:-train}" \
  --stream_eval \
  --prompt_len "${PROMPT_LEN:-1024}" \
  --continuation_len "${CONTINUATION_LEN:-32}" \
  --num_sequences "${NUM_EVAL:-128}" \
  --skip_sequences "${QUALITY_SKIP:-20000}" \
  --quant_configs "$configs" \
  --quality_risk_metric kl \
  "${common_quant[@]}" \
  --seed "${SEED:-0}" \
  --out_dir "$quality_out" \
  --wandb --wandb_project "${WANDB_PROJECT:-kv-reduce}" \
  --wandb_group "${WANDB_GROUP:-objective-shrinkage}" \
  --wandb_run_name "${RUN_TAG:-shrinkage}_quality"

python3 -u benchmark_spec_kv_quantization.py \
  --big_model "${BIG_MODEL:-Qwen/Qwen2.5-3B}" \
  --small_model "${SMALL_MODEL:-Qwen/Qwen2.5-1.5B}" \
  --big_device cuda:0 \
  --small_device cuda:0 \
  --big_dtype bf16 \
  --small_dtype bf16 \
  --attn_implementation sdpa \
  --dataset_name "${DATASET_NAME:-HuggingFaceFW/fineweb-edu}" \
  --dataset_config "${DATASET_CONFIG:-sample-10BT}" \
  --eval_split "${EVAL_SPLIT:-train}" \
  --stream_eval \
  --prompt_len "${PROMPT_LEN:-1024}" \
  --num_prompts "${NUM_EVAL:-128}" \
  --warmup_prompts "${WARMUP_PROMPTS:-2}" \
  --skip_prompts "${ACCEPTANCE_SKIP:-20200}" \
  --draft_steps "${DRAFT_STEPS:-4}" \
  --target_verification_mode sequential \
  --max_new_tokens "${MAX_NEW_TOKENS:-16}" \
  --quant_configs "$configs" \
  --target_quant_configs none \
  "${common_quant[@]}" \
  --seed "${SEED:-0}" \
  --out_dir "$acceptance_out" \
  --wandb --wandb_project "${WANDB_PROJECT:-kv-reduce}" \
  --wandb_group "${WANDB_GROUP:-objective-shrinkage}" \
  --wandb_run_name "${RUN_TAG:-shrinkage}_acceptance"

python3 -u aggregate_shrunk_allocator_eval.py \
  --bundle "$root/bundle.json" \
  --quality_dir "$quality_out" \
  --acceptance_dir "$acceptance_out" \
  --out_dir "$root/final_results"
