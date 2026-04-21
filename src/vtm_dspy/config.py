"""Centralized DSPy plus OpenRouter configuration for VTM."""

from __future__ import annotations

import os
from dataclasses import dataclass

from vtm.benchmarks.openrouter import (
    execution_model,
    openrouter_api_key,
    openrouter_base_url,
    rerank_model,
)

DEFAULT_DSPY_MODEL = "openrouter/nvidia/nemotron-3-super-120b-a12b:free"
DEFAULT_DSPY_MODEL_TYPE = "chat"


def _normalize_non_empty(value: str, *, field_name: str) -> str:
    resolved = value.strip()
    if not resolved:
        raise ValueError(f"{field_name} must not be empty")
    return resolved


def _normalize_openrouter_model(model_name: str) -> str:
    resolved = _normalize_non_empty(model_name, field_name="model_name")
    if resolved.startswith("openrouter/"):
        return resolved
    if resolved.startswith("openai/"):
        return f"openrouter/{resolved.removeprefix('openai/')}"
    return f"openrouter/{resolved}"


def resolve_dspy_model(explicit: str | None = None) -> str:
    """Resolve the configured DSPy model name in OpenRouter-prefixed form."""
    env_execution_model = os.getenv("VTM_EXECUTION_MODEL", "").strip()
    candidates = (
        explicit,
        os.getenv("VTM_DSPY_MODEL"),
        _normalize_openrouter_model(env_execution_model) if env_execution_model else None,
        DEFAULT_DSPY_MODEL,
    )
    for candidate in candidates:
        if candidate is not None and candidate.strip():
            return _normalize_openrouter_model(candidate)
    raise ValueError("unable to resolve a DSPy model name")


def resolve_dspy_lm_model(model_name: str) -> str:
    """Map the stored DSPy model id onto DSPy's OpenAI-compatible LM naming."""
    normalized = _normalize_openrouter_model(model_name)
    return f"openai/{normalized.removeprefix('openrouter/')}"


@dataclass(frozen=True)
class DSPyOpenRouterConfig:
    """Resolved OpenRouter settings shared by DSPy helpers and benchmark scripts."""

    base_url: str
    api_key: str | None
    execution_model: str
    rerank_model: str
    dspy_model: str
    model_type: str = DEFAULT_DSPY_MODEL_TYPE
    temperature: float | None = None
    max_tokens: int | None = None

    @classmethod
    def from_env(
        cls,
        *,
        base_url_value: str | None = None,
        api_key_value: str | None = None,
        execution_model_name: str | None = None,
        rerank_model_name: str | None = None,
        dspy_model_name: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> DSPyOpenRouterConfig:
        """Resolve one consistent config bundle from the repo's environment variables."""
        base_url = _normalize_non_empty(
            base_url_value or openrouter_base_url(),
            field_name="base_url",
        )
        api_key = (api_key_value if api_key_value is not None else openrouter_api_key()) or None
        execution = execution_model(execution_model_name)
        rerank = rerank_model(rerank_model_name)
        dspy_model = resolve_dspy_model(dspy_model_name)
        return cls(
            base_url=base_url,
            api_key=api_key,
            execution_model=execution,
            rerank_model=rerank,
            dspy_model=dspy_model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def require_api_key(self) -> str:
        """Return a configured API key or fail with a clear runtime error."""
        if self.api_key is None:
            raise ValueError(
                "OpenRouter API access requires OPENROUTER_API_KEY when running DSPy models"
            )
        return self.api_key

    def lm_model_name(self) -> str:
        """Return the DSPy LM id for OpenAI-compatible OpenRouter access."""
        return resolve_dspy_lm_model(self.dspy_model)

    def lm_kwargs(self) -> dict[str, object]:
        """Return keyword arguments suitable for `dspy.LM(...)`."""
        kwargs: dict[str, object] = {
            "api_base": self.base_url,
            "model_type": self.model_type,
        }
        if self.api_key is not None:
            kwargs["api_key"] = self.api_key
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        return kwargs

    def as_env(self) -> dict[str, str]:
        """Return the equivalent environment-variable mapping."""
        payload = {
            "VTM_OPENROUTER_BASE_URL": self.base_url,
            "VTM_EXECUTION_MODEL": self.execution_model,
            "VTM_RERANK_MODEL": self.rerank_model,
            "VTM_DSPY_MODEL": self.dspy_model,
        }
        if self.api_key is not None:
            payload["OPENROUTER_API_KEY"] = self.api_key
        return payload

    def summary(self) -> dict[str, str]:
        """Return a redacted summary suitable for dry-run logging."""
        return {
            "base_url": self.base_url,
            "execution_model": self.execution_model,
            "rerank_model": self.rerank_model,
            "dspy_model": self.dspy_model,
            "dspy_lm_model": self.lm_model_name(),
            "model_type": self.model_type,
            "api_key_configured": "true" if self.api_key is not None else "false",
        }


__all__ = [
    "DEFAULT_DSPY_MODEL",
    "DEFAULT_DSPY_MODEL_TYPE",
    "DSPyOpenRouterConfig",
    "resolve_dspy_lm_model",
    "resolve_dspy_model",
]
