#!/bin/bash
#SBATCH --job-name=kv_longbench_ret
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --exclude=catalyst-0-[9,15]
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
cd "${REPO_ROOT:-/home/sakshamg/KV_Reduce}"
mkdir -p logs outputs
if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

seed=$((10#${SLURM_ARRAY_TASK_ID:-0}))
examples="${LONGBENCH_EXAMPLES:-24}"
skip=$((seed * examples))
root="${OUT_ROOT:-outputs/kivi_longbench_passage_retrieval_v1/qwen25_15b}"
out_dir="$root/longbench_passage_retrieval/seed_${seed}"
model="${MODEL:-Qwen/Qwen2.5-1.5B}"
model_label="${MODEL_LABEL:-qwen25_15b}"
max_prompt_tokens="${MAX_PROMPT_TOKENS:-32752}"

export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

python3 -u eval_kv_quantized_multiple_choice.py \
  --model "$model" \
  --device cuda:0 \
  --attn_implementation sdpa \
  --task longbench_passage_retrieval \
  --num_examples "$examples" \
  --skip_examples "$skip" \
  --num_fewshot 0 \
  --max_prompt_tokens "$max_prompt_tokens" \
  --max_choice_tokens 8 \
  --seed "$seed" \
  --quant_configs "${QUANT_CONFIGS:-none;k8v4;k4v8;k4v4;k4v2;k2v4;k2v2}" \
  --key_quant_axis per_channel \
  --key_group_size 32 \
  --key_residual_length 128 \
  --value_quant_scheme affine \
  --out_dir "$out_dir" \
  --wandb \
  --wandb_project "${WANDB_PROJECT:-kv-reduce}" \
  --wandb_group kivi-longbench-retrieval \
  --wandb_run_name "${model_label}_longbench_retrieval_s${seed}"
