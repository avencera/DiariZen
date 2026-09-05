#!/usr/bin/env bash

set -euo pipefail

unset CONTAINER_API_KEY JUPYTER_TOKEN OPEN_BUTTON_TOKEN

recipe_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
venv_path="${DIARIZEN_VENV:-/root/code/diarizen-venv}"

cd "$recipe_dir"
"$venv_path/bin/python" prepare_full_corpus.py
"$venv_path/bin/python" prepare_voxconverse.py
exec ./run_full.sh
