#!/usr/bin/env bash
# Helper script to launch a vLLM OpenAI-compatible server on a self-hosted GPU box.

set -euo pipefail

# Activate virtual environment if it exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "../.venv/bin/activate" ]; then
    source ../.venv/bin/activate
fi

# Load .env file if it exists
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# Use GPU 3 by default if CUDA_VISIBLE_DEVICES is not set
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"3"}

MODEL_NAME=${VLLM_MODEL_NAME:-${VLLM_MODEL:-"meta-llama/Meta-Llama-3-8B-Instruct"}}
# MODEL_NAME=${VLLM_MODEL_NAME:-${VLLM_MODEL:-"Qwen/Qwen2.5-7B-Instruct"}}
HOST=${VLLM_HOST:-"0.0.0.0"}
PORT=${VLLM_PORT:-8000}
TP_SIZE=${VLLM_TP_SIZE:-1}
GPU_MEM_UTIL=${VLLM_GPU_MEM_UTIL:-0.80}
EXTRA_ARGS=${VLLM_EXTRA_ARGS:-""}
HF_TOKEN_FLAG=""

if [[ -n "${HF_TOKEN:-}" ]]; then
  HF_TOKEN_FLAG="--hf-token ${HF_TOKEN}"
fi

echo "[vLLM] Starting server for model ${MODEL_NAME} on ${HOST}:${PORT} (tp=${TP_SIZE})"
exec python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  ${HF_TOKEN_FLAG} ${EXTRA_ARGS}
