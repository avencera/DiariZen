#!/usr/bin/env bash

set -uo pipefail

unset CONTAINER_API_KEY JUPYTER_TOKEN OPEN_BUTTON_TOKEN

recipe_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
training_config="${DIARIZEN_TRAINING_CONFIG:-$recipe_dir/conf/full_wavlm_base_plus_16gb_upstream_v2.toml}"
config_experiment_id="$(basename "$training_config" .toml)"
experiment_id="${DIARIZEN_EXPERIMENT_ID:-$config_experiment_id}"
if [[ ! "$experiment_id" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo "invalid experiment ID: $experiment_id" >&2
    exit 2
fi
if [[ ! -f "$training_config" ]]; then
    echo "training configuration not found: $training_config" >&2
    exit 1
fi
if [[ "$experiment_id" != "$config_experiment_id" ]]; then
    echo "experiment ID must match the training configuration name: $config_experiment_id" >&2
    exit 2
fi

export DIARIZEN_EXPERIMENT_ID="$experiment_id"
export DIARIZEN_TRAINING_CONFIG="$training_config"

pipeline_log="$recipe_dir/$experiment_id.pipeline.log"
pipeline_pid_file="$recipe_dir/$experiment_id.pipeline.pid"
supervisor_log="$recipe_dir/$experiment_id.supervisor.log"
evaluation_log="$recipe_dir/$experiment_id.evaluation.log"
status_file="$recipe_dir/$experiment_id.status"
experiment_dir="$recipe_dir/exp_full/$experiment_id"
completion_marker="Training loop finished at epoch"
max_restarts=5
poll_seconds="${DIARIZEN_SUPERVISOR_POLL_SECONDS:-60}"

if [[ ! "$poll_seconds" =~ ^[0-9]+$ ]]; then
    echo "invalid supervisor poll interval: $poll_seconds" >&2
    exit 2
fi

log_status() {
    printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" >> "$supervisor_log"
}

write_status() {
    local status="$1"
    local temporary_status_file="$status_file.partial"

    printf '%s\n' "$status" > "$temporary_status_file"
    mv "$temporary_status_file" "$status_file"
}

wait_for_pipeline() {
    local pipeline_pid="$1"

    while kill -0 "$pipeline_pid" 2>/dev/null; do
        if (( poll_seconds > 0 )); then
            sleep "$poll_seconds"
        fi
    done
}

pipeline_log_offset() {
    if [[ -f "$pipeline_log" ]]; then
        wc -c < "$pipeline_log"
    else
        printf '0\n'
    fi
}

start_pipeline() {
    local -a command

    if [[ ! -f "$recipe_dir/data/full/provenance.json" \
        || ! -f "$recipe_dir/data/full/provenance.voxconverse.json" ]]; then
        command=("$recipe_dir/run_full_pipeline.sh")
    elif compgen -G "$experiment_dir/checkpoints/epoch_*" > /dev/null; then
        command=("$recipe_dir/run_full.sh" --resume)
    else
        command=("$recipe_dir/run_full.sh")
    fi

    # a prior completed run can remain in the append-only log
    pipeline_start_offset="$(pipeline_log_offset)"
    "${command[@]}" >> "$pipeline_log" 2>&1 &
    local pipeline_pid=$!
    printf '%s\n' "$pipeline_pid" > "$pipeline_pid_file"
    log_status "started pipeline PID $pipeline_pid: ${command[*]}"
    wait_for_pipeline "$pipeline_pid"
    wait "$pipeline_pid" || true
}

evaluate_completed_pipeline() {
    if [[ ! -f "$pipeline_log" ]]; then
        return 1
    fi
    if ! tail -c +$((pipeline_start_offset + 1)) "$pipeline_log" \
        | grep -F -- "$completion_marker" > /dev/null; then
        return 1
    fi

    for ((evaluation_attempt = 1; evaluation_attempt <= 3; evaluation_attempt++)); do
        log_status "starting evaluation attempt $evaluation_attempt"
        if "$recipe_dir/evaluate_full.sh" >> "$evaluation_log" 2>&1; then
            write_status "ready_to_stop"
            log_status "evaluation complete; GPU is ready to stop"
            return 0
        fi
    done

    log_status "evaluation failed after 3 attempts"
    write_status "failed"
    return 2
}

cd "$recipe_dir" || exit 1
write_status "running"
pipeline_start_offset="$(pipeline_log_offset)"
if [[ -f "$pipeline_pid_file" ]]; then
    initial_pid="$(cat "$pipeline_pid_file")"
    if [[ "$initial_pid" =~ ^[0-9]+$ ]] && kill -0 "$initial_pid" 2>/dev/null; then
        log_status "monitoring existing pipeline PID $initial_pid"
        wait_for_pipeline "$initial_pid"
    fi
fi

for ((attempt = 1; attempt <= max_restarts; attempt++)); do
    evaluate_completed_pipeline
    completion_status=$?
    if (( completion_status == 0 )); then
        exit 0
    elif (( completion_status == 2 )); then
        exit 1
    fi

    log_status "pipeline incomplete; restart attempt $attempt"
    start_pipeline

    evaluate_completed_pipeline
    completion_status=$?
    if (( completion_status == 0 )); then
        exit 0
    elif (( completion_status == 2 )); then
        exit 1
    fi
done

log_status "pipeline failed after $max_restarts restart attempts"
write_status "failed"
exit 1
