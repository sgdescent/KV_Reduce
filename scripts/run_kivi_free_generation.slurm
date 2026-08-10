#!/bin/bash
#SBATCH --job-name=kv_kivi_gen
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs outputs
source .venv/bin/activate

task_id="${SLURM_ARRAY_TASK_ID:-0}"
seed=$(( ${BASE_SEED:-0} + task_id ))
skip=$(( ${BASE_SKIP_PROMPTS:-0} + task_id * ${SKIP_STRIDE:-64} ))
tag="${MODEL_TAG:-qwen25_15b}"
out_dir="${OUT_ROOT:-outputs/kivi_free_generation}/${tag}/ctx_${PROMPT_LEN:-1024}/gen_${MAX_NEW_TOKENS:-64}/seed_${seed}"

cmd=(
  python3 -u eval_kv_quantized_generation.py
  --model "${MODEL:-Qwen/Qwen2.5-1.5B}"
  --device cuda:0
  --dtype "${DTYPE:-bf16}"
  --dataset_name "${DATASET_NAME:-HuggingFaceFW/fineweb-edu}"
  --dataset_config "${DATASET_CONFIG:-sample-10BT}"
  --split "${DATASET_SPLIT:-train}"
  --streaming
  --skip_prompts "$skip"
  --num_prompts "${NUM_PROMPTS:-32}"
  --prompt_len "${PROMPT_LEN:-1024}"
  --max_new_tokens "${MAX_NEW_TOKENS:-64}"
  --quant_configs "${QUANT_CONFIGS:-none;k8v4;k4v8;k4v4;k3v4;k4v3;k2v4;k4v2}"
  --key_quant_axis per_channel
  --key_group_size "${KEY_GROUP_SIZE:-32}"
  --key_residual_length "${KEY_RESIDUAL_LENGTH:-128}"
  --value_quant_scheme affine
  --seed "$seed"
  --out_dir "$out_dir"
)
if [[ "${ENABLE_WANDB:-1}" == "1" ]]; then
  cmd+=(
    --wandb
    --wandb_project "${WANDB_PROJECT:-kv-reduce}"
    --wandb_group "${WANDB_GROUP:-kivi-free-generation}"
    --wandb_run_name "${WANDB_RUN_NAME_PREFIX:-${tag}_free_gen}_c${PROMPT_LEN:-1024}_g${MAX_NEW_TOKENS:-64}_s${seed}"
  )
fi

printf 'Running command:'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
