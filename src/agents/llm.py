"""Utility helpers for obtaining the default LLM used across agents."""

import os
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv()


def get_llm(
    model: Optional[str] = None,
    temperature: float = 0.3,
    provider: Optional[str] = "vllm",
    **kwargs: Any
):
    """
    Return a LangChain-compatible client.
    Supports:
      - vLLM (via OpenAI protocol)
      - HuggingFace (local pipeline)
      - Google Gemini (fallback)

    Args:
        model: Model name to use.
        temperature: Temperature for sampling.
        provider: "vllm", "huggingface", "gemini", or None (defaults to vllm).
        **kwargs: Additional arguments (e.g., base_url for vLLM).
    """

    # 1) vLLM (default). If unavailable, fall back to Gemini (if key present).
    if provider == "vllm" or provider is None:
        llm = _build_vllm_client(model=model, temperature=temperature, **kwargs)
        if llm is not None:
            return llm
        gemini_llm = _build_gemini_client(model=model, temperature=temperature)
        if gemini_llm is not None:
            print("[get_llm] vLLM unavailable; falling back to Gemini")
            return gemini_llm
        print("Warning: vLLM unavailable and no Gemini credentials; returning None")
        return None

    # 2) Google Gemini (explicit provider)
    if provider == "gemini":
        return _build_gemini_client(model=model, temperature=temperature)

    # 3) HuggingFace local pipeline (explicit provider)
    if provider == "huggingface":
        return _build_huggingface_client(model=model, temperature=temperature, **kwargs)

    print("Warning: No valid LLM configuration found.")
    return None


def _build_vllm_client(model: Optional[str], temperature: float, **kwargs):
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        print("Warning: langchain-openai not installed. Install it via `pip install langchain-openai`.")
        return None

    model_name = model or os.getenv("VLLM_MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")

    # Default to localhost:8000 if not specified
    default_base_url = os.getenv("VLLM_API_BASE", "http://localhost:8000/v1")

    # Auto-select port based on model name if base_url is not explicitly provided
    if "base_url" not in kwargs:
        if "1B" in model_name or "1b" in model_name:
            default_base_url = "http://localhost:8001/v1"
        elif "8B" in model_name or "8b" in model_name:
            default_base_url = "http://localhost:8000/v1"

    base_url = kwargs.get("base_url", default_base_url)

    api_key = os.getenv("VLLM_API_KEY", "EMPTY")

    print(f"[get_llm] Using vLLM with model {model_name} at {base_url}")
    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        openai_api_key=api_key,
        openai_api_base=base_url,
        max_tokens=kwargs.get("max_tokens", None),
    )


def _build_gemini_client(model: Optional[str], temperature: float):
    available_gemini_models = [
        "gemini-2.0-flash-exp",
        "gemini-2.0-flash",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
    ]

    model_name = model or os.getenv("GOOGLE_MODEL_NAME") or available_gemini_models[0]
    gemini_api_key = os.getenv("GEMINI_API_KEY")

    if not gemini_api_key:
        print("Warning: GEMINI_API_KEY not set; cannot initialize Gemini client.")
        return None

    print(f"[get_llm] Using Google Gemini with model {model_name}")
    return _build_google_client(
        model_name=model_name,
        temperature=temperature,
        api_key=gemini_api_key,
    )


def _build_huggingface_client(model: Optional[str], temperature: float, **kwargs):
    try:
        from langchain_huggingface import HuggingFacePipeline
    except ImportError:
        print("Warning: langchain-huggingface not installed. Install it via `pip install langchain-huggingface`.")
        return None

    model_name = model or "meta-llama/Llama-3.2-1B"
    print(f"[get_llm] Using HuggingFace Local Pipeline with model {model_name}")

    # Note: This loads model into memory!
    return HuggingFacePipeline.from_model_id(
        model_id=model_name,
        task="text-generation",
        pipeline_kwargs={
            "max_new_tokens": kwargs.get("max_new_tokens", 1024),
            "temperature": temperature,
            "do_sample": True,
        },
    )


def _build_google_client(
    *,
    model_name: str,
    temperature: float,
    api_key: Optional[str] = None,
):
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError:
        print("Warning: langchain-google-genai is not installed. Install it via `pip install langchain-google-genai`.")
        return None

    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        google_api_key=api_key,
    )
