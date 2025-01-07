FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

COPY --from=ghcr.io/astral-sh/uv:0.5.7 /uv /uvx /bin/

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git build-essential

ADD . /app
WORKDIR /app

RUN uv sync
ENTRYPOINT uv run bentoml serve service:VLLM -p 50001
