#!/bin/bash
#SBATCH --job-name=kv_kivi_genagg
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --exclude=catalyst-0-[9,15]
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
source .venv/bin/activate

python3 -u aggregate_kv_quantized_generation.py \
  --summary_glob "${SUMMARY_GLOB:?Set SUMMARY_GLOB}" \
  --out_dir "${OUT_DIR:?Set OUT_DIR}" \
  --bootstrap_samples "${BOOTSTRAP_SAMPLES:-10000}"
