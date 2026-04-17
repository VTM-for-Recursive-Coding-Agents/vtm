"""Centralized OpenRouter defaults for maintained benchmark inference."""

from __future__ import annotations

import os

DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_EXECUTION_MODEL = "google/gemma-4-31b-it:free"
DEFAULT_RERANK_MODEL = "nvidia/nemotron-3-nano-30b-a3b:free"
OPTIONAL_STRONGER_ABLATION_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"


def openrouter_base_url() -> str:
    """Return the maintained OpenRouter base URL."""
    return os.getenv("VTM_OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_BASE_URL).strip()


def openrouter_api_key() -> str | None:
    """Return the OpenRouter API key when configured."""
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    return api_key or None


def execution_model(explicit: str | None = None) -> str:
    """Resolve the maintained execution model id."""
    resolved = (explicit or os.getenv("VTM_EXECUTION_MODEL") or DEFAULT_EXECUTION_MODEL).strip()
    if not resolved:
        raise ValueError("execution model id must not be empty")
    return resolved


def rerank_model(explicit: str | None = None) -> str:
    """Resolve the maintained rerank model id."""
    resolved = (explicit or os.getenv("VTM_RERANK_MODEL") or DEFAULT_RERANK_MODEL).strip()
    if not resolved:
        raise ValueError("rerank model id must not be empty")
    return resolved


__all__ = [
    "DEFAULT_EXECUTION_MODEL",
    "DEFAULT_OPENROUTER_BASE_URL",
    "DEFAULT_RERANK_MODEL",
    "OPTIONAL_STRONGER_ABLATION_MODEL",
    "execution_model",
    "openrouter_api_key",
    "openrouter_base_url",
    "rerank_model",
]
