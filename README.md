# ReviewerAgent

An on-premises paper reviewer agent that connects OpenReview human review data to a multi-agent pipeline based on RAG and LangGraph. It provides a Streamlit demo and evaluation scripts by self-hosting 1B/8B models with vLLM.
Repository: <https://github.com/suho-han/paper-review>

[[README.md in Korean]](docs/README_kor.md)

## Development Environments

VS Code with Copilot (mainly used Gemini 3 Pro)
Python 3.13.9
Deployment: Streamlit
OS: Linux (tested on Ubuntu 22.04+)
Package manager: uv (recommended), pip
GPU: NVIDIA RTX Pro 6000

## Quick Start (Streamlit Demo)

```bash
uv pip install -r requirements.txt
uv run streamlit run run_demo.py
```

- Default: Automatically launches 1B (port 8001, GPU mem util 0.5, max-len 4096) + 8B (port 8000, GPU mem util 0.7, max-len 6144) and opens the UI.
- To manually start the server, use `DEMO_SKIP_LLM_START=1` or `uv run streamlit run run_demo.py -- --skip-llm-start`.
- If a HuggingFace token is required, set `HF_TOKEN`.

### Example of Manual vLLM Launch

```bash
# 1B
uv run python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --port 8001 --gpu-memory-utilization 0.5 --max-model-len 4096 --tensor-parallel-size 1

# 8B (e.g., using GPU 2)
CUDA_VISIBLE_DEVICES=2 uv run python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --port 8000 --gpu-memory-utilization 0.7 --max-model-len 6144 --tensor-parallel-size 1

# Run the demo without automatically starting the LLM
DEMO_SKIP_LLM_START=1 uv run streamlit run run_demo.py
```

If issues arise, clean up residual processes with `uv run python scripts/kill_vllm.py` (or `bash scripts/kill_vllm.sh`).

## Data Pipeline

1. **Collection**

```bash
# Basic single paper collection
uv run src/data/data_collection.py

# Batch download (e.g., ICLR 2025)
uv run src/data/data_collection.py --batch --venue ICLR.cc/2025/Conference
```

1. **Vector DB Construction (ChromaDB)**

```bash
# Build the entire collection
uv run src/vectordb/build_all_vectordb.py

# Specific JSON → Specific Collection
uv run src/vectordb/build_vectordb.py --json_file data/ICLR.cc_2025_Conference_reviews.json --collection iclr_2025
```

1. **Search/Test**

```bash
# Basic test (all_reviews collection)
uv run src/vectordb/build_vectordb.py --test

# Test a specific collection
uv run src/vectordb/build_vectordb.py --test --collection iclr_2025 --query "What are the main weaknesses in experimental design?"
```

1. **Exploration/Visualization**: Use `notebooks/test_chromadb.ipynb` or `notebooks/test_agents.ipynb` to inspect collections and agent behavior.

## Evaluation Pipeline

- Test set: `data/evaluation/test_set_2025.json`
- Execution scripts: `src/evaluation/run_eval.py`, `src/evaluation/run_baseline_vs_agent.py`
- Key metrics: Weakness recall, rating correlation, hallucination rate

### Running Baseline vs Agent Experiments

The baseline runs **zero-RAG (Reviewer+Rating only, no search/ArXiv)**, while the agent executes the **CoordinatorAgent full workflow (RAG+ArXiv+Rating)** to compare metrics.

1. (Recommended) Check vLLM server status

```bash
uv run python scripts/healthcheck_vllm.py
```

1. Run the experiment (default: 25 samples)

```bash
uv run python src/evaluation/run_baseline_vs_agent.py \
  --data data/evaluation/test_set_2025.json \
  --sample 25 \
  --output outputs/results
```

1. Test on all samples (0)

```bash
uv run python src/evaluation/run_baseline_vs_agent.py \
  --data data/evaluation/test_set_2025.json \
  --sample 0 \
  --output outputs/results
```

1. Check the output

- If `--output` is a directory, results are saved as a bundle in the following paths:
  - `outputs/results/baseline_vs_agent/<UTC_TIMESTAMP>/report.json`
  - `outputs/results/baseline_vs_agent/<UTC_TIMESTAMP>/summary.md`
  - `outputs/results/baseline_vs_agent/<UTC_TIMESTAMP>/details.csv`

Options

- `--no-llm`: Skip LLM-based metrics (weakness recall/hallucination) and run without LLM servers
- `--output <path.json>`: Specify a JSON file path to save `report.json` at that location, along with `summary.md` and `details.csv` in the same folder

## Project Structure (Summary)

```plaintext
src/agents/          # Multi-agent based on LangGraph (parser, retriever, reviewer, rating, etc.)
src/data/            # OpenReview collection scripts
src/vectordb/        # ChromaDB build and test
src/evaluation/      # Quantitative evaluation scripts
scripts/             # vLLM management, health checks, monitoring utilities
notebooks/           # Notebooks for agent/DB experiments
data/, chromadb/     # Original data and vector DB storage
```

## Troubleshooting

- vLLM memory errors: Lower GPU mem util or reduce `--max-model-len`, and terminate residual processes.
- Port conflicts: Check occupancy with `lsof -i :8000` or `lsof -i :8001` and terminate.
- Demo stuck in retry loop: Force refresh the browser (Ctrl+Shift+R) to reset the Streamlit session state.

## References

- Check `FIXES.md` for vLLM default settings and recent updates.
- Refer to `src/agents/` for LangGraph pipeline and model execution flow.

### AI Review by [Stanford Agentic Reviewer](https://paperreview.ai/)

Go to [https://paperreview.ai/review](https://paperreview.ai/review) and paste the token below

```bash
vWuhcbzdBtWIOWRSWji03ZSGgJrD9A8jrKZgMG0HMA0
```

## TODO List

### Next

- [ ] Enhance Reviewer/Rating prompts (add few-shot examples, specify rubric)
- [ ] Improve RAG quality: Consider hybrid search, Cross-Encoder re-ranking
- [ ] Stabilize vLLM operations: Review healthcheck/alert thresholds, tune retry policies
