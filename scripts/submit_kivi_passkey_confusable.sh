#!/bin/bash
set -euo pipefail

export OUT_ROOT="${OUT_ROOT:-outputs/kivi_passkey_confusable_v3/qwen25_15b}"
export PASSKEY_NUM_CHOICES="${PASSKEY_NUM_CHOICES:-16}"
export PASSKEY_VARIANT=confusable_records
export PASSKEY_GENERATOR_VERSION=synthetic_associative_passkey_v3
export PASSKEY_WANDB_GROUP=kivi-passkey-confusable

exec bash scripts/submit_kivi_passkey_aggressive.sh
