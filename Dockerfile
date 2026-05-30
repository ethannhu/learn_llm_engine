FROM pytorch/pytorch:2.12.0-cuda13.0-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV HF_HOME=/workspace/.cache/huggingface
ENV TRANSFORMERS_CACHE=/workspace/.cache/huggingface
ENV TOKENIZERS_PARALLELISM=false

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    vim \
    nano \
    curl \
    wget \
    htop \
    tree \
    build-essential \
    ca-certificates \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

RUN python -m pip install --upgrade pip setuptools wheel

RUN pip install \
    transformers \
    safetensors \
    sentencepiece \
    accelerate \
    numpy \
    tqdm \
    einops \
    ipython

CMD ["/bin/bash"]