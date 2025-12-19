# Self-Hosted vLLM Setup Guide

This document captures the minimum steps for running ReviewerAgent against a locally hosted OSS LLM using [vLLM](https://github.com/vllm-project/vllm). The workflow mirrors the OpenAI-compatible API so LangChain can talk to it through the updated `get_llm` helper.

## 1. Prerequisites

- CUDA-ready GPU server (≥24 GB VRAM recommended for 8B models, ≥70 GB for 70B models)
- Python 3.10+ with `pip` or `uv`
- This repository's dependencies installed: `uv pip install -r requirements.txt`
- Enough disk space to store the target model weights (check the HuggingFace card for exact size)

## 2. Download or Mount the Model

```bash
# Example: pull Meta Llama 3 8B Instruct via huggingface-cli
ehco $HF_TOKEN | huggingface-cli login --token
huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct --local-dir /models/llama3-8b-instruct
```

You can substitute any other OSS model that vLLM supports (Qwen2, Mistral, etc.). Update `VLLM_MODEL_NAME` accordingly.

## 3. Start the vLLM OpenAI-Compatible Server

Use the unified Python launcher in `scripts/start_vllm.py`:

```bash
export VLLM_MODEL_NAME=/models/llama3-8b-instruct
export VLLM_HOST=0.0.0.0
export VLLM_PORT=8000
export VLLM_TP_SIZE=1          # set >1 when using multiple GPUs
export VLLM_GPU_MEM_UTIL=0.90  # lower if you encounter OOMs

python scripts/start_vllm.py start --model "$VLLM_MODEL_NAME" --port "$VLLM_PORT" --cuda 3 --tp-size "$VLLM_TP_SIZE" --gpu-mem-util "$VLLM_GPU_MEM_UTIL"
```

Presets are available:

```bash
# Llama 3.2 1B on port 8001
python scripts/start_vllm.py start-1b

# Meta-Llama 3 8B Instruct on port 8000
python scripts/start_vllm.py start-8b

# Start both (background, prints PIDs)
python scripts/start_vllm.py start-both
```

Additional flags can be passed via `--extra-args "..."` or environment `VLLM_EXTRA_ARGS` (e.g., `--max-model-len 8192`). If the model requires a HuggingFace token set `HF_TOKEN` before running the script.

## 4. Configure the ReviewerAgent to Use the Local Server

`src/agents/llm.py` now checks the following environment variables before falling back to Gemini:

| Variable | Description |
| --- | --- |
| `VLLM_API_BASE` | Base URL of the local OpenAI-compatible endpoint (e.g., `http://localhost:8000/v1`). |
| `VLLM_MODEL_NAME` | Model ID exposed by the server (typically matches `--model`). |
| `VLLM_API_KEY` | Optional API key. vLLM defaults to `EMPTY`, so this can stay unset. |
| `OPENAI_API_KEY` / `OPENAI_MODEL_NAME` | Used when calling the real OpenAI endpoint instead of vLLM. |

Example configuration for local testing:

```bash
export VLLM_API_BASE=http://localhost:8000/v1
export VLLM_MODEL_NAME=meta-llama/Meta-Llama-3-8B-Instruct
export VLLM_API_KEY=EMPTY
```

No code changes are required elsewhere; ReviewerAgent will pick up the local endpoint automatically.

## 5. Health Check

After the server is running, run a quick smoke test:

```bash
curl -X POST "$VLLM_BASE_URL/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${VLLM_API_KEY:-EMPTY}" \
  -d '{
        "model": "'$VLLM_MODEL_NAME'",
        "messages": [{"role":"user","content":"Hello"}]
      }'
```

You should receive a JSON completion. If not, check GPU utilization, logs, or adjust `VLLM_GPU_MEM_UTIL`.

## 6. Monitoring & Logging

- The arXiv agent cache/log now writes to `logs/arxiv_cache.json` and `logs/arxiv_queries.log`.
- To monitor LLM throughput, wrap `scripts/start_vllm_server.sh` with `systemd` or `supervisord` and capture stdout/stderr.

## 7. Next Steps

- Add multi-GPU tensor parallel configs for 70B models.
- Gate ReviewerAgent temperature/model choice via env vars or CLI flags.
- Integrate lightweight health checks before running large evaluation batches.
