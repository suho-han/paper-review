#!/usr/bin/env python3
"""
Unified vLLM launcher that replaces:
- scripts/start_vllm_server.sh
- scripts/start_llama_1b.sh
- scripts/start_llama_8b.sh
- scripts/start_both_llm.sh

Features:
- Start a single vLLM OpenAI-compatible server with environment defaults.
- Convenience commands for 1B and 8B presets.
- Start both 1B and 8B servers concurrently (backgrounded) and print PIDs.
- Loads .env (if present) similar to bash scripts.

Usage examples:
  python scripts/start_vllm.py start --model meta-llama/Meta-Llama-3-8B-Instruct --port 8000 --cuda 3
  python scripts/start_vllm.py start-1b
  python scripts/start_vllm.py start-8b
  python scripts/start_vllm.py start-both
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


def _load_dotenv_if_exists() -> None:
    """Lightweight .env loader (no external deps).

    - Looks for .env in CWD first, then parent directory (to mimic bash scripts).
    - Supports simple KEY=VALUE pairs; ignores comments and blank lines.
    - Respects existing environment (only sets if not already set).
    """

    def parse_env_file(path: Path) -> Dict[str, str]:
        out: Dict[str, str] = {}
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s or s.startswith("#"):
                        continue
                    if "=" not in s:
                        continue
                    k, v = s.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip("'\"")
                    if k:
                        out[k] = v
        except FileNotFoundError:
            pass
        return out

    cwd = Path.cwd()
    candidates = [cwd / ".env", cwd.parent / ".env"]
    for p in candidates:
        if p.exists():
            envs = parse_env_file(p)
            for k, v in envs.items():
                os.environ.setdefault(k, v)
            # First found .env wins, like the bash approach
            break


def _default(val: Optional[str], fallback: str) -> str:
    return fallback if (val is None or val == "") else val


def _build_server_args(
    model_name: str,
    host: str,
    port: int,
    tp_size: int,
    gpu_mem_util: float,
    extra_args: Optional[str] = None,
    hf_token: Optional[str] = None,
) -> List[str]:
    args: List[str] = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_name,
        "--host",
        host,
        "--port",
        str(port),
        "--tensor-parallel-size",
        str(tp_size),
        "--gpu-memory-utilization",
        str(gpu_mem_util),
    ]
    if hf_token:
        args.extend(["--hf-token", hf_token])
    if extra_args:
        args.extend(shlex.split(extra_args))
    return args


def _start_vllm_server(
    *,
    model_name: str,
    port: int,
    cuda_devices: str,
    gpu_mem_util: float,
    host: str = "0.0.0.0",
    tp_size: int = 1,
    extra_args: Optional[str] = None,
    hf_token: Optional[str] = None,
    start_in_background: bool = False,
) -> subprocess.Popen:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_devices

    cmd = _build_server_args(
        model_name=model_name,
        host=host,
        port=port,
        tp_size=tp_size,
        gpu_mem_util=gpu_mem_util,
        extra_args=extra_args,
        hf_token=hf_token,
    )

    print(
        f"[vLLM] Starting server for model {model_name} on {host}:{port} (tp={tp_size}, CUDA_VISIBLE_DEVICES={cuda_devices}, gpu_mem={gpu_mem_util})"
    )
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=None,
        stderr=None,
        start_new_session=start_in_background,  # detach when backgrounding
    )
    return proc


def cmd_start(args: argparse.Namespace) -> None:
    _load_dotenv_if_exists()

    # Environment defaults (compatible with the original bash scripts)
    model_name = _default(
        os.environ.get("VLLM_MODEL_NAME") or os.environ.get("VLLM_MODEL"),
        "meta-llama/Meta-Llama-3-8B-Instruct",
    )
    host = _default(os.environ.get("VLLM_HOST"), "0.0.0.0")
    port = int(_default(os.environ.get("VLLM_PORT"), "8000"))
    tp_size = int(_default(os.environ.get("VLLM_TP_SIZE"), "1"))
    gpu_mem_util = float(_default(os.environ.get("VLLM_GPU_MEM_UTIL"), "0.8"))
    cuda_devices = _default(os.environ.get("CUDA_VISIBLE_DEVICES"), "3")
    extra_args = os.environ.get("VLLM_EXTRA_ARGS", "") or None
    hf_token = os.environ.get("HF_TOKEN") or None

    # Apply CLI overrides if provided
    if args.model:
        model_name = args.model
    if args.port is not None:
        port = args.port
    if args.cuda is not None:
        cuda_devices = args.cuda
    if args.gpu_mem_util is not None:
        gpu_mem_util = args.gpu_mem_util
    if args.host is not None:
        host = args.host
    if args.tp_size is not None:
        tp_size = args.tp_size
    if args.extra_args:
        extra_args = args.extra_args

    proc = _start_vllm_server(
        model_name=model_name,
        port=port,
        cuda_devices=cuda_devices,
        gpu_mem_util=gpu_mem_util,
        host=host,
        tp_size=tp_size,
        extra_args=extra_args,
        hf_token=hf_token,
        start_in_background=False,
    )
    # Foreground run: wait for the process to exit
    proc.wait()


def cmd_start_1b(args: argparse.Namespace) -> None:
    _load_dotenv_if_exists()

    model_name = _default(
        os.environ.get("VLLM_MODEL_NAME") or os.environ.get("VLLM_MODEL"),
        "meta-llama/Llama-3.2-1B",
    )
    port = int(_default(os.environ.get("VLLM_PORT_1B") or os.environ.get("VLLM_PORT"), "8001"))
    cuda_devices = _default(os.environ.get("CUDA_VISIBLE_DEVICES_1B") or os.environ.get("CUDA_VISIBLE_DEVICES"), "2")
    gpu_mem_util = float(_default(os.environ.get("VLLM_GPU_MEM_UTIL_1B") or os.environ.get("VLLM_GPU_MEM_UTIL"), "0.6"))
    host = _default(os.environ.get("VLLM_HOST"), "0.0.0.0")
    tp_size = int(_default(os.environ.get("VLLM_TP_SIZE"), "1"))
    extra_args = os.environ.get("VLLM_EXTRA_ARGS", "") or None
    hf_token = os.environ.get("HF_TOKEN") or None

    # Apply CLI overrides
    if args.model:
        model_name = args.model
    if args.port is not None:
        port = args.port
    if args.cuda is not None:
        cuda_devices = args.cuda
    if args.gpu_mem_util is not None:
        gpu_mem_util = args.gpu_mem_util

    proc = _start_vllm_server(
        model_name=model_name,
        port=port,
        cuda_devices=cuda_devices,
        gpu_mem_util=gpu_mem_util,
        host=host,
        tp_size=tp_size,
        extra_args=extra_args,
        hf_token=hf_token,
        start_in_background=False,
    )
    proc.wait()


def cmd_start_8b(args: argparse.Namespace) -> None:
    _load_dotenv_if_exists()

    model_name = _default(
        os.environ.get("VLLM_MODEL_NAME_8B")
        or os.environ.get("VLLM_MODEL_NAME")
        or os.environ.get("VLLM_MODEL"),
        "meta-llama/Meta-Llama-3-8B-Instruct",
    )
    port = int(_default(os.environ.get("VLLM_PORT_8B") or os.environ.get("VLLM_PORT"), "8000"))
    cuda_devices = _default(os.environ.get("CUDA_VISIBLE_DEVICES_8B") or os.environ.get("CUDA_VISIBLE_DEVICES"), "3")
    gpu_mem_util = float(_default(os.environ.get("VLLM_GPU_MEM_UTIL_8B") or os.environ.get("VLLM_GPU_MEM_UTIL"), "0.8"))
    host = _default(os.environ.get("VLLM_HOST"), "0.0.0.0")
    tp_size = int(_default(os.environ.get("VLLM_TP_SIZE"), "1"))
    extra_args = os.environ.get("VLLM_EXTRA_ARGS", "") or None
    hf_token = os.environ.get("HF_TOKEN") or None

    # Apply CLI overrides
    if args.model:
        model_name = args.model
    if args.port is not None:
        port = args.port
    if args.cuda is not None:
        cuda_devices = args.cuda
    if args.gpu_mem_util is not None:
        gpu_mem_util = args.gpu_mem_util

    proc = _start_vllm_server(
        model_name=model_name,
        port=port,
        cuda_devices=cuda_devices,
        gpu_mem_util=gpu_mem_util,
        host=host,
        tp_size=tp_size,
        extra_args=extra_args,
        hf_token=hf_token,
        start_in_background=False,
    )
    proc.wait()


def cmd_start_both(args: argparse.Namespace) -> None:
    _load_dotenv_if_exists()

    # 1B defaults
    model_1b = _default(
        os.environ.get("VLLM_MODEL_NAME") or os.environ.get("VLLM_MODEL"),
        "meta-llama/Llama-3.2-1B",
    )
    port_1b = int(_default(os.environ.get("VLLM_PORT_1B"), "8001"))
    cuda_1b = _default(os.environ.get("CUDA_VISIBLE_DEVICES_1B"), "2")
    mem_1b = float(_default(os.environ.get("VLLM_GPU_MEM_UTIL_1B"), "0.6"))

    # 8B defaults
    model_8b = _default(
        os.environ.get("VLLM_MODEL_NAME_8B")
        or os.environ.get("VLLM_MODEL_NAME")
        or os.environ.get("VLLM_MODEL"),
        "meta-llama/Meta-Llama-3-8B-Instruct",
    )
    port_8b = int(_default(os.environ.get("VLLM_PORT_8B"), "8000"))
    cuda_8b = _default(os.environ.get("CUDA_VISIBLE_DEVICES_8B"), "3")
    mem_8b = float(_default(os.environ.get("VLLM_GPU_MEM_UTIL_8B"), "0.8"))

    host = _default(os.environ.get("VLLM_HOST"), "0.0.0.0")
    tp_size = int(_default(os.environ.get("VLLM_TP_SIZE"), "1"))
    extra_args = os.environ.get("VLLM_EXTRA_ARGS", "") or None
    hf_token = os.environ.get("HF_TOKEN") or None

    # CLI overrides
    if args.port_1b is not None:
        port_1b = args.port_1b
    if args.port_8b is not None:
        port_8b = args.port_8b
    if args.cuda_1b is not None:
        cuda_1b = args.cuda_1b
    if args.cuda_8b is not None:
        cuda_8b = args.cuda_8b

    # Start both in background like the original bash script
    p1 = _start_vllm_server(
        model_name=model_1b,
        port=port_1b,
        cuda_devices=cuda_1b,
        gpu_mem_util=mem_1b,
        host=host,
        tp_size=tp_size,
        extra_args=extra_args,
        hf_token=hf_token,
        start_in_background=True,
    )
    p8 = _start_vllm_server(
        model_name=model_8b,
        port=port_8b,
        cuda_devices=cuda_8b,
        gpu_mem_util=mem_8b,
        host=host,
        tp_size=tp_size,
        extra_args=extra_args,
        hf_token=hf_token,
        start_in_background=True,
    )

    print(
        f"Started Llama 1B (PID: {p1.pid}) on port {port_1b}, GPU {cuda_1b}"
    )
    print(
        f"Started Llama 8B (PID: {p8.pid}) on port {port_8b}, GPU {cuda_8b}"
    )
    print(
        f"Wait for logs to confirm readiness (vLLM HTTP ready). Use 'ps -p {p1.pid},{p8.pid}' to check processes."
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Unified vLLM launcher")
    sub = p.add_subparsers(dest="cmd", required=True)

    # Generic start
    s = sub.add_parser("start", help="Start a single vLLM server (foreground)")
    s.add_argument("--model", type=str, help="Model name (HF repo id)")
    s.add_argument("--port", type=int, help="HTTP port")
    s.add_argument("--cuda", type=str, help="CUDA_VISIBLE_DEVICES value")
    s.add_argument("--gpu-mem-util", type=float, help="GPU memory utilization [0,1]")
    s.add_argument("--host", type=str, help="Bind host (default 0.0.0.0)")
    s.add_argument("--tp-size", type=int, help="Tensor parallel size")
    s.add_argument("--extra-args", type=str, help="Extra args appended to vLLM command")
    s.set_defaults(func=cmd_start)

    # 1B preset (foreground)
    s1 = sub.add_parser("start-1b", help="Start Llama-3.2-1B on port 8001 (foreground)")
    s1.add_argument("--model", type=str, help="Override model name")
    s1.add_argument("--port", type=int, help="Override port")
    s1.add_argument("--cuda", type=str, help="Override CUDA_VISIBLE_DEVICES")
    s1.add_argument("--gpu-mem-util", type=float, help="Override GPU memory utilization")
    s1.set_defaults(func=cmd_start_1b)

    # 8B preset (foreground)
    s8 = sub.add_parser("start-8b", help="Start Meta-Llama-3-8B-Instruct on port 8000 (foreground)")
    s8.add_argument("--model", type=str, help="Override model name")
    s8.add_argument("--port", type=int, help="Override port")
    s8.add_argument("--cuda", type=str, help="Override CUDA_VISIBLE_DEVICES")
    s8.add_argument("--gpu-mem-util", type=float, help="Override GPU memory utilization")
    s8.set_defaults(func=cmd_start_8b)

    # both (background)
    sb = sub.add_parser("start-both", help="Start 1B (8001) and 8B (8000) in background")
    sb.add_argument("--port-1b", type=int, help="Port for 1B (default 8001)")
    sb.add_argument("--port-8b", type=int, help="Port for 8B (default 8000)")
    sb.add_argument("--cuda-1b", type=str, help="CUDA for 1B (default 2)")
    sb.add_argument("--cuda-8b", type=str, help="CUDA for 8B (default 3)")
    sb.set_defaults(func=cmd_start_both)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
