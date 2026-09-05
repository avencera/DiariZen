#!/usr/bin/env bash

set -euo pipefail

unset CONTAINER_API_KEY JUPYTER_TOKEN OPEN_BUTTON_TOKEN

recipe_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
venv_path="${DIARIZEN_VENV:-/root/code/diarizen-venv}"
experiment_id="${DIARIZEN_EXPERIMENT_ID:-full_wavlm_base_plus_16gb_upstream_v2}"
if [[ ! "$experiment_id" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo "invalid experiment ID: $experiment_id" >&2
    exit 2
fi

experiment_dir="$recipe_dir/exp_full/$experiment_id"
ranked_index="$experiment_dir/ranked_checkpoints/index.json"
embedding_model="$recipe_dir/artifacts/wespeaker-voxceleb-resnet34-LM-pyannote.bin"
dscore_dir="${DSCORE_DIR:-/root/code/dscore}"
inference_profile="upstream-constrained-ahc-v2-input-bound"
config_file="$(find "$experiment_dir" -maxdepth 1 -name 'config__*.toml' -print | sort | tail -n 1)"

source "$recipe_dir/nvidia_driver_compat.sh"

if [[ -z "$config_file" ]]; then
    echo "no saved training configuration found in $experiment_dir" >&2
    exit 1
fi

cd "$recipe_dir/../diar_ssl"
for dataset in AMI AliMeeting AISHELL4 VoxConverse; do
    data_dir="../speakrs/data/full/test/$dataset"
    output_dir="$experiment_dir/inference/$inference_profile/best5/test/$dataset"
    "$venv_path/bin/python" infer_avg.py \
        --configuration "$config_file" \
        --in_wav_scp "$data_dir/wav.scp" \
        --out_dir "$output_dir" \
        --embedding_model "$embedding_model" \
        --ranked_checkpoint_index "$ranked_index" \
        --avg_ckpt_num 5 \
        --seg_duration 8 \
        --batch_size 16 \
        --apply_median_filtering \
        --clustering_method AgglomerativeClustering \
        --min_speakers 1 \
        --max_speakers 20 \
        --ahc_threshold 0.70 \
        --min_cluster_size 30

    "$venv_path/bin/python" "$dscore_dir/score.py" \
        -u "$data_dir/all.uem" \
        -r "$data_dir/rttm" \
        -s "$output_dir"/*.rttm \
        --collar 0 \
        > "$output_dir/result_collar0.txt"
done
