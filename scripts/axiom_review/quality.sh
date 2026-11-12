#!/usr/bin/env bash
# One of the six October review quality settings. No scheduler submission here.
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"
task=${1:-${SLURM_ARRAY_TASK_ID:-0}}
if [[ ! "$task" =~ ^[0-5]$ ]]; then echo 'Task must be 0..5.' >&2; exit 2; fi
if (( $# > 2 )) || [[ ${2:-} != '' && ${2:-} != '--dry-run' ]]; then echo 'Usage: quality.sh [0..5] [--dry-run]' >&2; exit 2; fi
residual=$(( (task % 2) * 128 ))
seed=$(( (task / 2) % 2 ))
context=1024
if (( task >= 4 )); then context=16384; seed=0; fi
out="${OUT_ROOT:-outputs/axiom-review}/axiom_quality_ctx${context}_r${residual}_s${seed}"
if [[ -e "$out" ]]; then echo "Refusing to overwrite $out" >&2; exit 1; fi
export TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_XET=1 WANDB_MODE=disabled
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
cmd=("${PYTHON:-python}" -u -m kv_reduce.experiments.benchmark_spec_kv_quantization
  --big_model Qwen/Qwen2.5-3B --small_model Qwen/Qwen2.5-1.5B
  --big_device cuda:0 --small_device cuda:0 --attn_implementation sdpa
  --dataset_name HuggingFaceFW/fineweb-edu --dataset_config sample-10BT
  --eval_split train --stream_eval --shuffle_eval --seed "$seed"
  --prompt_len "$context" --num_prompts 24 --warmup_prompts 1 --skip_prompts 100
  --max_new_tokens 256 --draft_steps 4 --target_verification_mode sequential
  --quant_configs 'none;k8v4;k4v8;k4v4' --key_quant_axis per_channel
  --key_group_size 32 --key_residual_length "$residual" --value_quant_scheme affine
  --out_dir "$out")
if [[ ${2:-} == '--dry-run' ]]; then printf '%q ' "${cmd[@]}"; printf '\n'; exit 0; fi
exec "${cmd[@]}"
