#!/usr/bin/env bash
# One of two packed-prototype settings; known batched correctness limitations apply.
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"
task=${1:-${SLURM_ARRAY_TASK_ID:-0}}
if [[ ! "$task" =~ ^[0-1]$ ]]; then echo 'Task must be 0 or 1.' >&2; exit 2; fi
if (( $# > 2 )) || [[ ${2:-} != '' && ${2:-} != '--dry-run' ]]; then echo 'Usage: packed.sh [0|1] [--dry-run]' >&2; exit 2; fi
context=1024; prompts=24
if (( task == 1 )); then context=16384; prompts=12; fi
out="${OUT_ROOT:-outputs/axiom-review}/axiom_packed_ctx${context}"
if [[ -e "$out" ]]; then echo "Refusing to overwrite $out" >&2; exit 1; fi
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_XET=1 WANDB_MODE=disabled
cmd=("${PYTHON:-python}" -u -m kv_reduce.experiments.packed_review --context "$context" --prompts "$prompts" --tokens 256 --out "$out")
if [[ ${2:-} == '--dry-run' ]]; then printf '%q ' "${cmd[@]}"; printf '\n'; exit 0; fi
exec "${cmd[@]}"
