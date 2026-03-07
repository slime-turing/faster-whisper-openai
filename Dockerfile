FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ARG http_proxy
ARG https_proxy
ARG no_proxy
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    HF_HOME=/var/cache/stt/hf \
    XDG_CACHE_HOME=/var/cache/stt/xdg \
    STT_CACHE_DIR=/var/cache/stt \
    STT_TMP_DIR=/var/tmp/stt \
    TRANSFORMERS_CACHE=/var/cache/stt/hf \
    http_proxy=${http_proxy} \
    https_proxy=${https_proxy} \
    no_proxy=${no_proxy} \
    HTTP_PROXY=${HTTP_PROXY} \
    HTTPS_PROXY=${HTTPS_PROXY} \
    NO_PROXY=${NO_PROXY}

RUN apt-get -o Acquire::Retries=5 update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    ca-certificates \
    curl \
    tini \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install -r requirements.txt

COPY app /app/app

RUN mkdir -p /var/cache/stt/hf /var/cache/stt/xdg /var/tmp/stt

EXPOSE 9000

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9000", "--workers", "1"]
