from __future__ import annotations

import os
from typing import Any

from vtm.adapters.embeddings import EmbeddingAdapter


class OpenAIEmbeddingAdapter:
    def __init__(
        self,
        *,
        model: str,
        client: Any | None = None,
        api_key: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
    ) -> None:
        if not model:
            raise ValueError("OpenAI embedding adapter requires a non-empty model name")
        self._model = model
        self._client = client or self._build_client(api_key=api_key, api_key_env=api_key_env)

    @property
    def adapter_id(self) -> str:
        return f"openai:{self._model}"

    def embed_text(self, text: str) -> tuple[float, ...]:
        response = self._client.embeddings.create(
            model=self._model,
            input=text,
        )
        data = getattr(response, "data", None)
        if not isinstance(data, list) or not data:
            raise ValueError("OpenAI embedding adapter returned no embedding data")
        embedding = getattr(data[0], "embedding", None)
        if not isinstance(embedding, list) or not all(
            isinstance(value, (int, float)) for value in embedding
        ):
            raise ValueError("OpenAI embedding adapter returned an invalid embedding")
        return tuple(float(value) for value in embedding)

    def _build_client(self, *, api_key: str | None, api_key_env: str) -> Any:
        resolved_api_key = api_key or os.getenv(api_key_env)
        if not resolved_api_key:
            raise ValueError(
                "OpenAI embedding adapter requires api_key or the environment variable "
                f"{api_key_env!r}"
            )
        try:
            from openai import OpenAI  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "OpenAI embedding adapter requires the optional 'openai' dependency"
            ) from exc
        return OpenAI(api_key=resolved_api_key)


__all__ = ["EmbeddingAdapter", "OpenAIEmbeddingAdapter"]
