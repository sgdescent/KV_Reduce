#!/bin/bash
set -euo pipefail

cd "${REPO_ROOT:-/home/sakshamg/KV_Reduce}"
mkdir -p logs
exclude="${EXCLUDE_NODES:-catalyst-0-9,catalyst-0-15}"

prep=$(sbatch --parsable \
  --exclude="$exclude" \
  --export=ALL \
  scripts/prepare_shrunk_allocator.slurm)

eval_args=(
  --parsable
  --dependency="afterok:$prep"
  --exclude="$exclude"
  --export=ALL
)
if [[ "${START_HELD:-1}" == "1" ]]; then
  eval_args+=(--hold)
fi
evaluation=$(sbatch "${eval_args[@]}" scripts/run_shrunk_allocator_eval.slurm)

printf 'Submitted shrinkage-aware allocator campaign\n  prepare: %s\n  evaluate: %s\n  output: %s\n' \
  "$prep" "$evaluation" "${OUT_ROOT:?Set OUT_ROOT}"
