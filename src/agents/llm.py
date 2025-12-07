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

    # 1. vLLM (OpenAI Compatible) - Default
    if provider == "vllm" or provider is None:
        return _build_vllm_client(model=model, temperature=temperature, **kwargs)

    # 2. HuggingFace Local Pipeline
    if provider == "huggingface":
        return _build_huggingface_client(model=model, temperature=temperature, **kwargs)

    # 3. Google Gemini
    if provider == "gemini":
        # Define a list of preferred models in order of priority
        available_gemini_models = [
            "gemini-2.0-flash-exp",
            "gemini-2.0-flash",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
        ]

        # Determine which model to use:
        google_model = model or os.getenv("GOOGLE_MODEL_NAME") or available_gemini_models[0]
        gemini_api_key = os.getenv("GEMINI_API_KEY")

        if not gemini_api_key:
            print("Warning: provider='gemini' requested but GEMINI_API_KEY is not set.")
            return None

        print(f"[get_llm] Using Google Gemini with model {google_model}")
        return _build_google_client(
            model_name=google_model,
            temperature=temperature,
            api_key=gemini_api_key,
        )

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
