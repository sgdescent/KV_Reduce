#!/bin/bash
#SBATCH --job-name=kv_passkey_layer
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

task_id=$((10#${SLURM_ARRAY_TASK_ID:-0}))
model_idx=$((task_id / 3))
seed=$((task_id % 3))
case "$model_idx" in
  0) model="Qwen/Qwen2.5-1.5B"; model_tag="qwen25_15b"; num_layers=28 ;;
  1) model="meta-llama/Llama-3.2-3B"; model_tag="llama32_3b"; num_layers=28 ;;
  2) model="Qwen/Qwen3-4B"; model_tag="qwen3_4b"; num_layers=36 ;;
  *) echo "Unsupported model index: $model_idx" >&2; exit 2 ;;
esac

context="${PASSKEY_CONTEXT:-16384}"
examples="${PASSKEY_EXAMPLES:-32}"
# Powered 16K runs use source indices 0--191. Keep this localization sweep
# disjoint by default so it cannot tune conclusions on the replication set.
skip_base="${PASSKEY_SKIP_BASE:-192}"
skip=$((skip_base + seed * examples))
root="${OUT_ROOT:-outputs/passkey_layer_sensitivity_v1}/$model_tag"
allocation_dir="$root/allocations"
out_dir="$root/ctx_${context}/seed_${seed}"

python3 -u prepare_passkey_layer_sensitivity.py \
  --num_layers "$num_layers" \
  --layers "${LAYERS:-top:8}" \
  --base_k_bits "${BASE_K_BITS:-4}" \
  --base_v_bits "${BASE_V_BITS:-4}" \
  --reduced_bits "${REDUCED_BITS:-2}" \
  --out_dir "$allocation_dir" >/dev/null
quant_configs="$(<"$allocation_dir/quant_configs.txt")"

export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

python3 -u eval_kv_quantized_multiple_choice.py \
  --model "$model" \
  --device cuda:0 \
  --attn_implementation sdpa \
  --task passkey \
  --num_examples "$examples" \
  --skip_examples "$skip" \
  --num_fewshot 0 \
  --max_prompt_tokens "$context" \
  --max_choice_tokens 16 \
  --passkey_num_choices 16 \
  --passkey_variant confusable_records \
  --passkey_score raw \
  --seed "$seed" \
  --quant_configs "$quant_configs" \
  --key_quant_axis per_channel \
  --key_group_size 32 \
  --key_residual_length 128 \
  --value_quant_scheme affine \
  --out_dir "$out_dir" \
  --wandb \
  --wandb_project "${WANDB_PROJECT:-kv-reduce}" \
  --wandb_group "${WANDB_GROUP:-passkey-layer-sensitivity}" \
  --wandb_run_name "${model_tag}_passkey_layer_c${context}_s${seed}"
