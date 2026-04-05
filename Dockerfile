# syntax=docker/dockerfile:1.4
# OmniVoice TTS HTTP API (GPU). NVIDIA Container Toolkit; BuildKit для кэша pip.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-venv \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/pip \
    python3.10 -m pip install -U pip setuptools wheel && \
    python3.10 -m pip install \
    torch==2.8.0+cu128 \
    torchaudio==2.8.0+cu128 \
    --extra-index-url https://download.pytorch.org/whl/cu128

COPY OmniVoice /app/OmniVoice

RUN --mount=type=cache,target=/root/.cache/pip \
    python3.10 -m pip install --no-cache-dir -e /app/OmniVoice

COPY api /app/api

RUN --mount=type=cache,target=/root/.cache/pip \
    python3.10 -m pip install --no-cache-dir -r /app/api/requirements-api.txt

ENV OMNIVOICE_MODEL=k2-fsa/OmniVoice \
    HF_HOME=/root/.cache/huggingface

EXPOSE 38080

CMD ["python3.10", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "38080"]