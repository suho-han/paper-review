import os
import shlex
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import httpx


@dataclass
class ServerSpec:
    name: str
    model: str
    port: int
    cuda_visible_devices: str
    gpu_mem_util: float
    host: str = "0.0.0.0"
    tp_size: int = 1
    extra_args: Optional[str] = None
    hf_token: Optional[str] = None


@dataclass
class ServerHandle:
    spec: ServerSpec
    pid: Optional[int]
    started: bool
    already_running: bool


def _is_ready(port: int, timeout_s: float = 0.0) -> bool:
    url = f"http://localhost:{port}/v1/models"
    try:
        with httpx.Client(timeout=timeout_s or 1.0) as client:
            resp = client.get(url)
            return resp.status_code == 200
    except Exception:
        return False


def _wait_ready(port: int, timeout_s: float = 120.0, poll_interval: float = 0.5) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if _is_ready(port):
            return True
        time.sleep(poll_interval)
    return _is_ready(port)


def _start_server(spec: ServerSpec) -> ServerHandle:
    if _is_ready(spec.port):
        return ServerHandle(spec=spec, pid=None, started=False, already_running=True)

    cmd: List[str] = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        spec.model,
        "--host",
        spec.host,
        "--port",
        str(spec.port),
        "--tensor-parallel-size",
        str(spec.tp_size),
        "--gpu-memory-utilization",
        str(spec.gpu_mem_util),
    ]

    if spec.hf_token:
        cmd.extend(["--hf-token", spec.hf_token])

    if spec.extra_args:
        cmd.extend(shlex.split(spec.extra_args))

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = spec.cuda_visible_devices

    # Inherit stdout/stderr so logs are visible in the terminal
    proc = os.spawnve(os.P_NOWAIT, sys.executable, cmd, env)

    return ServerHandle(spec=spec, pid=proc, started=True, already_running=False)


def start_default_servers(wait: bool = True) -> Dict[str, ServerHandle]:
    # Defaults mirror the existing bash scripts
    spec_1b = ServerSpec(
        name="1B",
        model=os.environ.get("VLLM_MODEL_NAME", os.environ.get("VLLM_MODEL", "meta-llama/Llama-3.2-1B")),
        port=int(os.environ.get("VLLM_PORT_1B", 8001)),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES_1B", "3"),
        gpu_mem_util=float(os.environ.get("VLLM_GPU_MEM_UTIL_1B", 0.6)),
        host=os.environ.get("VLLM_HOST", "0.0.0.0"),
        tp_size=int(os.environ.get("VLLM_TP_SIZE", 1)),
        extra_args=os.environ.get("VLLM_EXTRA_ARGS", ""),
        hf_token=os.environ.get("HF_TOKEN"),
    )

    spec_8b = ServerSpec(
        name="8B",
        model=os.environ.get("VLLM_MODEL_NAME_8B", os.environ.get("VLLM_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")),
        port=int(os.environ.get("VLLM_PORT_8B", 8000)),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES_8B", "2"),
        gpu_mem_util=float(os.environ.get("VLLM_GPU_MEM_UTIL_8B", 0.8)),
        host=os.environ.get("VLLM_HOST", "0.0.0.0"),
        tp_size=int(os.environ.get("VLLM_TP_SIZE", 1)),
        extra_args=os.environ.get("VLLM_EXTRA_ARGS", ""),
        hf_token=os.environ.get("HF_TOKEN"),
    )

    handles: Dict[str, ServerHandle] = {}

    for spec in (spec_1b, spec_8b):
        handle = _start_server(spec)
        handles[spec.name] = handle

    if wait:
        for name, h in handles.items():
            if h.already_running:
                continue
            ready = _wait_ready(h.spec.port, timeout_s=300.0)
            if not ready:
                raise RuntimeError(f"vLLM server '{name}' on port {h.spec.port} failed to become ready in time")

    return handles


def ensure_port_ready(port: int, timeout_s: float = 120.0) -> bool:
    return _wait_ready(port, timeout_s=timeout_s)
