"""Minimal OpenAI-compatible chat client used by optional adapters."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib import error, request


@dataclass(frozen=True)
class OpenAICompatibleChatConfig:
    """Connection settings for an OpenAI-compatible chat endpoint."""

    base_url: str
    api_key: str
    timeout_seconds: int = 180
    extra_body: Mapping[str, Any] | None = None


class OpenAICompatibleChatClient:
    """Tiny JSON-over-HTTP client for OpenAI-compatible chat APIs."""

    def __init__(self, config: OpenAICompatibleChatConfig) -> None:
        """Validate and store endpoint configuration."""
        if not config.base_url.strip():
            raise ValueError("OpenAI-compatible chat client requires a non-empty base_url")
        self._config = config

    def create_chat_completion(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: int = 8192,
        seed: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Issue a chat-completions request and return the decoded JSON body."""
        payload: dict[str, Any] = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if seed is not None:
            payload["seed"] = seed
        if response_format is not None:
            payload["response_format"] = response_format
        if self._config.extra_body is not None:
            payload.update(dict(self._config.extra_body))

        body = json.dumps(payload).encode("utf-8")
        http_request = request.Request(
            self._chat_endpoint(self._config.base_url),
            data=body,
            headers={
                "Authorization": f"Bearer {self._config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with request.urlopen(http_request, timeout=self._config.timeout_seconds) as response:
                raw_text = response.read().decode("utf-8", errors="replace")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"OpenAI-compatible chat request failed with HTTP {exc.code}: {detail}"
            ) from exc
        except error.URLError as exc:
            raise RuntimeError(f"OpenAI-compatible chat request failed: {exc.reason}") from exc
        try:
            raw_payload = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            preview = self._response_preview(raw_text)
            raise RuntimeError(
                "OpenAI-compatible chat response was not valid JSON. "
                f"preview={preview}"
            ) from exc
        if not isinstance(raw_payload, dict):
            raise RuntimeError("OpenAI-compatible chat response must be a JSON object")
        return raw_payload

    def extract_message_text(self, response_payload: dict[str, Any]) -> str:
        """Extract assistant text from a chat-completions style response."""
        choices = response_payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("OpenAI-compatible chat response contained no choices")
        choice = choices[0]
        if not isinstance(choice, dict):
            raise RuntimeError("OpenAI-compatible chat response contained no choice object")
        message = choice.get("message")
        if not isinstance(message, dict):
            raise RuntimeError("OpenAI-compatible chat response contained no message object")
        content = self._coerce_text(message.get("content"))
        if content:
            return content
        refusal = self._coerce_text(message.get("refusal"))
        if refusal:
            return refusal
        choice_text = self._coerce_text(choice.get("text"))
        if choice_text:
            return choice_text
        detail = json.dumps(
            {
                "choice_keys": sorted(choice.keys()),
                "message_keys": sorted(message.keys()),
                "content_type": type(message.get("content")).__name__,
            },
            sort_keys=True,
        )
        raise RuntimeError(
            "OpenAI-compatible chat response contained unsupported content. "
            f"shape={detail}"
        )

    def _coerce_text(self, value: Any) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            collected = [self._coerce_text(item) for item in value]
            return "".join(part for part in collected if part)
        if isinstance(value, dict):
            item_type = value.get("type")
            if isinstance(item_type, str) and item_type not in {
                "text",
                "output_text",
                "message",
                "content",
            }:
                nested = self._coerce_text(value.get("content"))
                if nested:
                    return nested
                nested = self._coerce_text(value.get("output_text"))
                if nested:
                    return nested
                return ""
            text = value.get("text")
            if isinstance(text, str):
                return text
            if isinstance(text, dict):
                nested = self._coerce_text(text)
                if nested:
                    return nested
            for key in ("output_text", "content", "value"):
                nested = self._coerce_text(value.get(key))
                if nested:
                    return nested
        return ""

    def _chat_endpoint(self, base_url: str) -> str:
        normalized = base_url.rstrip("/")
        if normalized.endswith("/chat/completions"):
            return normalized
        if normalized.endswith("/v1"):
            return f"{normalized}/chat/completions"
        return f"{normalized}/v1/chat/completions"

    def _response_preview(self, raw_text: str, *, limit: int = 400) -> str:
        compact = " ".join(raw_text.split())
        if len(compact) <= limit:
            return compact
        return compact[: max(0, limit - 3)].rstrip() + "..."


__all__ = ["OpenAICompatibleChatClient", "OpenAICompatibleChatConfig"]
