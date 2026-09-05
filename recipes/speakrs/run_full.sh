#!/usr/bin/env bash

set -euo pipefail

unset CONTAINER_API_KEY JUPYTER_TOKEN OPEN_BUTTON_TOKEN

recipe_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
venv_path="${DIARIZEN_VENV:-/root/code/diarizen-venv}"
training_config="${DIARIZEN_TRAINING_CONFIG:-$recipe_dir/conf/full_wavlm_base_plus_16gb.toml}"
resume_args=()

source "$recipe_dir/nvidia_driver_compat.sh"

if [[ ! -f "$recipe_dir/data/full/provenance.voxconverse.json" ]]; then
    echo "VoxConverse preparation is incomplete" >&2
    exit 1
fi

if [[ ! -f "$training_config" ]]; then
    echo "training configuration not found: $training_config" >&2
    exit 1
fi

if [[ "${1:-}" == "--resume" ]]; then
    resume_args=(--resume)
elif [[ -n "${1:-}" ]]; then
    echo "usage: $0 [--resume]" >&2
    exit 2
fi

cd "$recipe_dir/../diar_ssl"
exec "$venv_path/bin/accelerate" launch \
    --num_processes 1 \
    --mixed_precision bf16 \
    run_dual_opt.py \
    --configuration "$training_config" \
    --mode train \
    "${resume_args[@]}"
