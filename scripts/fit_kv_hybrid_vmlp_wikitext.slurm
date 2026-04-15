#!/bin/bash
#SBATCH --job-name=kv_hybrid_vmlp
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --partition=general
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Runs fit_kv_hybrid_vmlp_probe_fast_eval.py on one node with two GPUs (cuda:0 / cuda:1).
#
# Before submitting:
#   1) Create a logs directory: mkdir -p logs
#   2) Export your W&B API key so the job inherits it (recommended):
#        export WANDB_API_KEY="your_key_here"
#      Or run `wandb login` once in the same conda env on a login node (writes ~/.netrc).
#   3) Activate your Python environment below.
#   4) Adjust #SBATCH --partition if your cluster uses a different queue (e.g. facunix).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# Optional: conda
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate KV_Reduce

python fit_kv_hybrid_vmlp_probe_fast_eval.py \
  --big_model Qwen/Qwen2.5-3B-Instruct \
  --small_model Qwen/Qwen2.5-1.5B-Instruct \
  --big_device cuda:0 \
  --small_device cuda:1 \
  --big_dtype bf16 \
  --small_dtype bf16 \
  --dataset_name wikitext \
  --dataset_config wikitext-2-raw-v1 \
  --train_split train \
  --eval_split validation \
  --seq_len 256 \
  --train_sequences 100000000 \
  --eval_sequences 64 \
  --position_stride 4 \
  --shuffle_train \
  --lambda_reg 1e-4 \
  --v_rows_per_layer 100000 \
  --v_hidden_dim 512 \
  --v_dropout 0.0 \
  --v_epochs 10 \
  --v_batch_size 1024 \
  --v_lr 3e-4 \
  --v_weight_decay 1e-4 \
  --v_cos_loss_weight 0.1 \
  --v_val_frac 0.05 \
  --out_dir outputs/qwen25_3b_to_15b_hybrid_vmlp_alltrain_v1 \
  --wandb \
  --wandb_project kv-reduce \
  --wandb_run_name "qwen25_3b_15b_wikitext_${SLURM_JOB_ID:-local}"
