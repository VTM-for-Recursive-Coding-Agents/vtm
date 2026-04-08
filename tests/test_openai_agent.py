from __future__ import annotations

import json

from vtm.adapters.openai_agent import OpenAICompatibleAgentModelAdapter
from vtm.adapters.openai_chat import OpenAICompatibleChatClient, OpenAICompatibleChatConfig
from vtm.agents import AgentConversationMessage, AgentModelTurnRequest, AgentToolSpec


class FakeChatClient:
    def __init__(self) -> None:
        self.last_kwargs: dict[str, object] | None = None

    def create_chat_completion(self, **kwargs: object) -> dict[str, object]:
        self.last_kwargs = dict(kwargs)
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {"assistant_message": "done", "done": True}
                        )
                    }
                }
            ]
        }

    def extract_message_text(self, response_payload: dict[str, object]) -> str:
        choices = response_payload["choices"]
        assert isinstance(choices, list)
        message = choices[0]["message"]
        assert isinstance(message, dict)
        return str(message["content"])


def test_openai_agent_adapter_forwards_sampling_controls() -> None:
    client = FakeChatClient()
    adapter = OpenAICompatibleAgentModelAdapter(model="gpt-test", client=client)
    request = AgentModelTurnRequest(
        mode="benchmark_autonomous",
        prompt_profile="vtm-native-agent-v1",
        workspace="/tmp/workspace",
        task_payload={"attempt_index": 2},
        messages=(AgentConversationMessage(role="user", content="Fix it"),),
        tools=(AgentToolSpec(name="terminal", description="Run a command"),),
        sampling_temperature=0.4,
        sampling_seed=1234,
    )

    response = adapter.complete_turn(request)

    assert response.done is True
    assert client.last_kwargs is not None
    assert client.last_kwargs["temperature"] == 0.4
    assert client.last_kwargs["seed"] == 1234


def test_openai_chat_client_only_includes_seed_when_present(monkeypatch) -> None:
    captured_bodies: list[dict[str, object]] = []

    class FakeHttpResponse:
        def __enter__(self) -> FakeHttpResponse:
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode("utf-8")

    def fake_urlopen(http_request, timeout):
        del timeout
        captured_bodies.append(json.loads(http_request.data.decode("utf-8")))
        return FakeHttpResponse()

    monkeypatch.setattr("vtm.adapters.openai_chat.request.urlopen", fake_urlopen)
    client = OpenAICompatibleChatClient(
        OpenAICompatibleChatConfig(
            base_url="http://127.0.0.1:8000",
            api_key="test",
        )
    )

    client.create_chat_completion(model="gpt-test", messages=[{"role": "user", "content": "hi"}])
    client.create_chat_completion(
        model="gpt-test",
        messages=[{"role": "user", "content": "hi"}],
        seed=42,
    )

    assert "seed" not in captured_bodies[0]
    assert captured_bodies[1]["seed"] == 42
