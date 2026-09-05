#!/usr/bin/env bash

set -euo pipefail

unset CONTAINER_API_KEY JUPYTER_TOKEN OPEN_BUTTON_TOKEN

recipe_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
venv_path="${DIARIZEN_VENV:-/root/code/diarizen-venv}"
official_model_dir="$recipe_dir/artifacts/diarizen-meeting-base"
official_model="$official_model_dir/pytorch_model.bin"
embedding_model="$recipe_dir/artifacts/wespeaker-voxceleb-resnet34-LM-pyannote.bin"
dscore_dir="${DSCORE_DIR:-/root/code/dscore}"
output_root="$recipe_dir/exp_controls/diarizen-meeting-base/model-card-v3-engine-bound"
expected_config_sha256="3b0a6c7c308477b127f7908d2c1b3dc15e23db4b58992dfdc5b38bed841aa859"
expected_model_sha256="9c4c4ee09ed5e5ab0982fe732e44268079fbad8adb3d69acfc4517c6448974e9"
expected_embedding_sha256="366edf44f4c80889a3eb7a9d7bdf02c4aede3127f7dd15e274dcdb826b143c56"
der_tolerance="1.0"
control_failed=0

source "$recipe_dir/nvidia_driver_compat.sh"

if [[ ! -f "$official_model_dir/config.toml" || ! -f "$official_model" || ! -f "$embedding_model" ]]; then
    echo "official meeting-base model is incomplete: $official_model_dir" >&2
    exit 1
fi

actual_config_sha256="$(sha256sum "$official_model_dir/config.toml" | cut -d' ' -f1)"
actual_model_sha256="$(sha256sum "$official_model" | cut -d' ' -f1)"
actual_embedding_sha256="$(sha256sum "$embedding_model" | cut -d' ' -f1)"
if [[ "$actual_config_sha256" != "$expected_config_sha256" \
    || "$actual_model_sha256" != "$expected_model_sha256" \
    || "$actual_embedding_sha256" != "$expected_embedding_sha256" ]]; then
    echo "official control artifact hash does not match" >&2
    exit 1
fi

cd "$recipe_dir/../diar_ssl"
for dataset in AMI AliMeeting AISHELL4; do
    data_dir="../speakrs/data/full/test/$dataset"
    output_dir="$output_root/test/$dataset"
    "$venv_path/bin/python" infer_avg.py \
        --configuration "$official_model_dir/config.toml" \
        --in_wav_scp "$data_dir/wav.scp" \
        --out_dir "$output_dir" \
        --embedding_model "$embedding_model" \
        --segmentation_model "$official_model" \
        --seg_duration 8 \
        --batch_size 16 \
        --no-apply_median_filtering \
        --clustering_method AgglomerativeClustering \
        --min_speakers 2 \
        --max_speakers 8 \
        --ahc_threshold 0.70 \
        --min_cluster_size 30

    "$venv_path/bin/python" "$dscore_dir/score.py" \
        -u "$data_dir/all.uem" \
        -r "$data_dir/rttm" \
        -s "$output_dir"/*.rttm \
        --collar 0 \
        > "$output_dir/result_collar0.txt"

    case "$dataset" in
        AMI) expected_der="15.6" ;;
        AliMeeting) expected_der="17.7" ;;
        AISHELL4) expected_der="12.0" ;;
    esac
    if ! "$venv_path/bin/python" "$recipe_dir/verify_der_result.py" \
        --result "$output_dir/result_collar0.txt" \
        --expected "$expected_der" \
        --tolerance "$der_tolerance"; then
        control_failed=1
    fi
done

exit "$control_failed"
