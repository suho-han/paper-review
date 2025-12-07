import argparse
import asyncio
import json
import time
from typing import Dict, List

import aiohttp
import numpy as np


async def send_request(session, url, prompt, max_tokens, model_name):
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": False
    }
    start_time = time.time()
    try:
        async with session.post(url, json=payload) as response:
            if response.status != 200:
                text = await response.text()
                print(f"Request failed: {response.status} - {text}")
                return None
            result = await response.json()
            end_time = time.time()
            return {
                "latency": end_time - start_time,
                "usage": result.get("usage", {}),
                "output": result["choices"][0]["message"]["content"]
            }
    except Exception as e:
        print(f"Request error: {e}")
        return None


async def benchmark(args):
    url = f"{args.base_url}/v1/chat/completions"
    prompts = [
        "Summarize the concept of attention mechanisms in deep learning.",
        "Explain the difference between transformer and RNN.",
        "Write a review for a paper about large language models.",
        "What is the capital of France?",
        "Generate a python script to sort a list."
    ] * args.num_requests

    print(f"Starting benchmark on {url} with {len(prompts)} requests...")
    print(f"Model: {args.model}")

    async with aiohttp.ClientSession() as session:
        tasks = [send_request(session, url, p, args.max_tokens, args.model) for p in prompts]
        start_total = time.time()
        results = await asyncio.gather(*tasks)
        end_total = time.time()

    valid_results = [r for r in results if r is not None]

    if not valid_results:
        print("All requests failed.")
        return

    latencies = [r["latency"] for r in valid_results]
    total_tokens = sum(r["usage"].get("completion_tokens", 0) for r in valid_results)
    total_duration = end_total - start_total

    print(f"\nBenchmark Results:")
    print(f"Total Requests: {len(valid_results)}")
    print(f"Total Duration: {total_duration:.2f}s")
    print(f"Average Latency: {np.mean(latencies):.2f}s")
    print(f"P95 Latency: {np.percentile(latencies, 95):.2f}s")
    print(f"Total Output Tokens: {total_tokens}")
    print(f"Throughput: {total_tokens / total_duration:.2f} tokens/s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark vLLM Server")
    parser.add_argument("--base-url", type=str, default="http://localhost:8000", help="vLLM server base URL")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="Model name to use")
    parser.add_argument("--num-requests", type=int, default=2, help="Number of times to repeat the prompt set")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens to generate")
    args = parser.parse_args()
    asyncio.run(benchmark(args))
