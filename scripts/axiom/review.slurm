#!/usr/bin/env bash
#SBATCH --job-name=axiom_review
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
# Submit from the repository root; create logs/ before sbatch.
# Add your site's partition/account options to sbatch.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
case "${1:-quality}" in
  quality) exec bash scripts/axiom/quality.sh "${SLURM_ARRAY_TASK_ID:?Use --array=0-5%2}" ;;
  packed) exec bash scripts/axiom/packed.sh "${SLURM_ARRAY_TASK_ID:?Use --array=0-1%1}" ;;
  *) echo 'Mode must be quality or packed.' >&2; exit 2 ;;
esac
