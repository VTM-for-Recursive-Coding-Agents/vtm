"""Minimal OpenAI-compatible chat client used by optional adapters."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib import error, request


@dataclass(frozen=True)
class OpenAICompatibleChatConfig:
    """Connection settings for an OpenAI-compatible chat endpoint."""

    base_url: str
    api_key: str
    timeout_seconds: int = 180


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
                raw_payload = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"OpenAI-compatible chat request failed with HTTP {exc.code}: {detail}"
            ) from exc
        if not isinstance(raw_payload, dict):
            raise RuntimeError("OpenAI-compatible chat response must be a JSON object")
        return raw_payload

    def extract_message_text(self, response_payload: dict[str, Any]) -> str:
        """Extract assistant text from a chat-completions style response."""
        choices = response_payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("OpenAI-compatible chat response contained no choices")
        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise RuntimeError("OpenAI-compatible chat response contained no message object")
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            collected = [
                str(item.get("text", ""))
                for item in content
                if isinstance(item, dict) and item.get("type") in {None, "text"}
            ]
            return "".join(collected)
        raise RuntimeError("OpenAI-compatible chat response contained unsupported content")

    def _chat_endpoint(self, base_url: str) -> str:
        normalized = base_url.rstrip("/")
        if normalized.endswith("/chat/completions"):
            return normalized
        if normalized.endswith("/v1"):
            return f"{normalized}/chat/completions"
        return f"{normalized}/v1/chat/completions"


__all__ = ["OpenAICompatibleChatClient", "OpenAICompatibleChatConfig"]
