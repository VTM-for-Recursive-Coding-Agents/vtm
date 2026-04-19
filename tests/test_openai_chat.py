from __future__ import annotations

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
