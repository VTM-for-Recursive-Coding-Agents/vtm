from __future__ import annotations

import json
from urllib import error

import pytest

from vtm.adapters.openai_chat import OpenAICompatibleChatClient, OpenAICompatibleChatConfig


def _client() -> OpenAICompatibleChatClient:
    return OpenAICompatibleChatClient(
        OpenAICompatibleChatConfig(
            base_url="https://openrouter.example/api/v1",
            api_key="openrouter-test-key",
        )
    )


def test_extract_message_text_supports_output_text_parts() -> None:
    client = _client()

    text = client.extract_message_text(
        {
            "choices": [
                {
                    "message": {
                        "content": [
                            {"type": "reasoning", "text": "ignored"},
                            {"type": "output_text", "text": "```python\nprint(42)\n```"},
                        ]
                    }
                }
            ]
        }
    )

    assert text == "```python\nprint(42)\n```"


def test_extract_message_text_supports_nested_text_value() -> None:
    client = _client()

    text = client.extract_message_text(
        {
            "choices": [
                {
                    "message": {
                        "content": [
                            {"type": "output_text", "text": {"value": "hello"}},
                            {"type": "text", "text": {"text": " world"}},
                        ]
                    }
                }
            ]
        }
    )

    assert text == "hello world"


def test_extract_message_text_supports_nested_content_objects() -> None:
    client = _client()

    text = client.extract_message_text(
        {
            "choices": [
                {
                    "message": {
                        "content": {
                            "type": "message",
                            "content": [
                                {
                                    "type": "output_text",
                                    "content": {"type": "text", "text": "print(42)"},
                                }
                            ],
                        }
                    }
                }
            ]
        }
    )

    assert text == "print(42)"


def test_extract_message_text_falls_back_to_choice_text() -> None:
    client = _client()

    text = client.extract_message_text(
        {
            "choices": [
                {
                    "text": "plain completion text",
                    "message": {"content": None},
                }
            ]
        }
    )

    assert text == "plain completion text"


def test_extract_message_text_supports_message_refusal() -> None:
    client = _client()

    text = client.extract_message_text(
        {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "refusal": "I can't help with that.",
                    }
                }
            ]
        }
    )

    assert text == "I can't help with that."


def test_create_chat_completion_raises_helpful_error_for_non_json_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _client()

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            del exc_type, exc, tb

        def read(self) -> bytes:
            return b"<html>temporary upstream gateway issue</html>"

    monkeypatch.setattr("vtm.adapters.openai_chat.request.urlopen", lambda *args, **kwargs: FakeResponse())

    with pytest.raises(RuntimeError, match="not valid JSON") as excinfo:
        client.create_chat_completion(
            model="qwen/qwen3-coder-next",
            messages=[{"role": "user", "content": "hello"}],
        )

    assert "temporary upstream gateway issue" in str(excinfo.value)


def test_create_chat_completion_wraps_url_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client()

    def raise_url_error(*args, **kwargs):
        del args, kwargs
        raise error.URLError("connection reset")

    monkeypatch.setattr("vtm.adapters.openai_chat.request.urlopen", raise_url_error)

    with pytest.raises(RuntimeError, match="connection reset"):
        client.create_chat_completion(
            model="qwen/qwen3-coder-next",
            messages=[{"role": "user", "content": "hello"}],
        )


def test_create_chat_completion_merges_extra_body(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}
    client = OpenAICompatibleChatClient(
        OpenAICompatibleChatConfig(
            base_url="https://openrouter.example/api/v1",
            api_key="openrouter-test-key",
            extra_body={
                "provider": {
                    "only": ["ionstream/fp8"],
                    "allow_fallbacks": False,
                }
            },
        )
    )

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            del exc_type, exc, tb

        def read(self) -> bytes:
            return json.dumps(
                {"choices": [{"message": {"content": "hello"}}]}
            ).encode("utf-8")

    def fake_urlopen(http_request, timeout):
        del timeout
        captured["payload"] = json.loads(http_request.data.decode("utf-8"))
        return FakeResponse()

    monkeypatch.setattr("vtm.adapters.openai_chat.request.urlopen", fake_urlopen)

    response = client.create_chat_completion(
        model="qwen/qwen3-coder-next",
        messages=[{"role": "user", "content": "hello"}],
    )

    assert response["choices"][0]["message"]["content"] == "hello"
    assert captured["payload"] == {
        "model": "qwen/qwen3-coder-next",
        "temperature": 0.0,
        "max_tokens": 8192,
        "messages": [{"role": "user", "content": "hello"}],
        "provider": {
            "only": ["ionstream/fp8"],
            "allow_fallbacks": False,
        },
    }
