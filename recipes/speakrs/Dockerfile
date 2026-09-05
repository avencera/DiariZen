# Linux AMD64 training runtime for the Speakrs WavLM Large recipe.
# Contains code and dependencies only. No datasets, checkpoints, or credentials.

FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.10
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    LANG=C.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-venv \
        python3.10-dev \
        python3-pip \
        ca-certificates \
        curl \
        ffmpeg \
        git \
        build-essential \
        libsndfile1 \
        sox \
        && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/diarizen
COPY . /opt/diarizen

RUN python3.10 -m pip install --upgrade pip \
    && python3.10 -m pip install --no-cache-dir \
        --index-url ${TORCH_INDEX} \
        "torch==2.11.0" \
        "torchaudio==2.11.0" \
    && python3.10 -m pip install --no-cache-dir \
        "numpy==1.26.4" \
        "accelerate==1.6.0" \
        "torchmetrics==1.9.0" \
        "toml" \
        "soundfile" \
        "scipy" \
        "pandas" \
        "tqdm" \
        "einops" \
        "pytest" \
        "ruff" \
    && python3.10 -m pip install --no-cache-dir "torchcodec==0.13.0" \
    && python3.10 -m pip install --no-cache-dir -e /opt/diarizen/pyannote-audio \
    && python3.10 -m pip install --no-cache-dir -e /opt/diarizen \
    && find /opt/diarizen -name '.git' -prune -o -name '*.pt' -print -o -name '*.flac' -print -o -name '*.wav' -print | head

WORKDIR /opt/diarizen
CMD ["python3.10", "recipes/speakrs/large_run.py", "--help"]
