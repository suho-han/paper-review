#!/usr/bin/env bash
# Start Llama-3.2-1B on Port 8001

# Set defaults
export VLLM_MODEL_NAME="meta-llama/Llama-3.2-1B"
export VLLM_PORT=8001
# Use a different GPU if available, or share the same one (might OOM if not careful)
# Defaulting to GPU 2 to avoid conflict with 8B on GPU 3, but user should adjust.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"2"} 
export VLLM_GPU_MEM_UTIL=0.6

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Run the generic vLLM startup script
bash "${SCRIPT_DIR}/start_vllm_server.sh"
