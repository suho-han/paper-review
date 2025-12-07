#!/usr/bin/env bash
# Start Llama-3.1-8B-Instruct on Port 8000

# Set defaults
export VLLM_MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"
export VLLM_PORT=8000
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"3"}
export VLLM_GPU_MEM_UTIL=0.8

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Run the generic vLLM startup script
bash "${SCRIPT_DIR}/start_vllm_server.sh"
