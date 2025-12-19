#!/usr/bin/env bash
# Kill all vLLM server processes

echo "Killing all vLLM processes..."

# Find and kill all python processes running vllm.entrypoints.openai.api_server
pkill -f "vllm.entrypoints.openai.api_server"

# Give processes time to terminate
sleep 2

# Force kill if still running
pkill -9 -f "vllm.entrypoints.openai.api_server"

echo "All vLLM processes terminated."
echo "You can now restart the servers."
