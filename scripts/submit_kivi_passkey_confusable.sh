#!/bin/bash
set -euo pipefail

export MODEL="${MODEL:-Qwen/Qwen2.5-1.5B}"
export MODEL_TAG="${MODEL_TAG:-qwen25_15b}"
export OUT_ROOT="${OUT_ROOT:-outputs/kivi_passkey_confusable_v3/$MODEL_TAG}"
export PASSKEY_NUM_CHOICES="${PASSKEY_NUM_CHOICES:-16}"
export PASSKEY_VARIANT=confusable_records
export PASSKEY_SCORE=normalized
export PASSKEY_GENERATOR_VERSION=synthetic_associative_passkey_v3
export PASSKEY_WANDB_GROUP="${PASSKEY_WANDB_GROUP:-kivi-passkey-confusable}"

exec bash scripts/submit_kivi_passkey_aggressive.sh
