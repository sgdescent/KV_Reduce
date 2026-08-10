#!/bin/bash
#SBATCH --job-name=kv_kivi_rqual
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
cd "${REPO_ROOT:-/home/sakshamg/KV_Reduce}"
mkdir -p logs outputs
if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

task_id="${SLURM_ARRAY_TASK_ID:-0}"
domain_id=$((task_id / 3))
seed=$((task_id % 3))
case "$domain_id" in
  0)
    domain=c4
    dataset=allenai/c4
    config=en
    split=validation
    column=text
    context=1024
    sequences="${C4_SEQUENCES:-48}"
    ;;
  1)
    domain=gsm8k
    dataset=openai/gsm8k
    config=main
    split=test
    column=question
    context=512
    sequences="${GSM8K_SEQUENCES:-48}"
    ;;
  2)
    domain=humaneval
    dataset=openai/openai_humaneval
    config=openai_humaneval
    split=test
    column=prompt
    context=512
    sequences="${HUMANEVAL_SEQUENCES:-24}"
    ;;
  *)
    echo "Unsupported domain id $domain_id" >&2
    exit 2
    ;;
esac

root="${QUALITY_ROOT:-outputs/kivi_objective_robustness/quality}"
out_dir="$root/$domain/ctx_${context}/seed_${seed}"
configs="${QUANT_CONFIGS:-none;k8v4;k4v8;k4v4;k3v4;k4v3;k2v4;k4v2;k2v2}"

export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

python3 -u profile_kv_quality_sensitivity.py \
  --model Qwen/Qwen2.5-1.5B \
  --device cuda:0 \
  --attn_implementation sdpa \
  --dataset_name "$dataset" \
  --dataset_config "$config" \
  --text_column "$column" \
  --eval_split "$split" \
  --stream_eval \
  --shuffle_eval \
  --prompt_len "$context" \
  --continuation_len "${CONTINUATION_LEN:-32}" \
  --num_sequences "$sequences" \
  --seed "$seed" \
  --quant_configs "$configs" \
  --key_quant_axis per_channel \
  --key_group_size 32 \
  --key_residual_length 128 \
  --value_quant_scheme affine \
  --out_dir "$out_dir" \
  --wandb \
  --wandb_project "${WANDB_PROJECT:-kv-reduce}" \
  --wandb_group kivi-objective-robustness-quality \
  --wandb_run_name "${domain}_kivi_quality_c${context}_s${seed}"
