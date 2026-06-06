#!/bin/bash
#SBATCH --job-name=kv_high_savings
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/kv_high_savings-%j.out
#SBATCH --error=logs/kv_high_savings-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

if [ -d ".venv" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

BIG_MODEL="${BIG_MODEL:-Qwen/Qwen2.5-7B}"
SMALL_MODEL="${SMALL_MODEL:-Qwen/Qwen2.5-3B}"
CONTEXTS="${CONTEXTS:-1024,4096,8192,16384,32768,65536}"
DRAFT_TAIL_LEN="${DRAFT_TAIL_LEN:-4}"
TARGET_QUANT_BITS="${TARGET_QUANT_BITS:-8,4,3}"
TARGET_MLA_LATENT_DIMS="${TARGET_MLA_LATENT_DIMS:-128,256,512}"
TARGET_MLA_ROPE_DIM="${TARGET_MLA_ROPE_DIM:-64}"
PLOT_CONTEXT="${PLOT_CONTEXT:-32768}"
OUT_DIR="${OUT_DIR:-outputs/high_savings_design_space}"

python3 estimate_high_savings_design_space.py \
  --big_model "$BIG_MODEL" \
  --small_model "$SMALL_MODEL" \
  --contexts "$CONTEXTS" \
  --draft_tail_len "$DRAFT_TAIL_LEN" \
  --target_quant_bits "$TARGET_QUANT_BITS" \
  --target_mla_latent_dims "$TARGET_MLA_LATENT_DIMS" \
  --target_mla_rope_dim "$TARGET_MLA_ROPE_DIM" \
  --plot_context "$PLOT_CONTEXT" \
  --out_dir "$OUT_DIR"
