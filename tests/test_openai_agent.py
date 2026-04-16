from __future__ import annotations

import json

from vtm.adapters.openai_agent import OpenAICompatibleAgentModelAdapter
from vtm.adapters.openai_chat import OpenAICompatibleChatClient, OpenAICompatibleChatConfig
from vtm.agents import AgentConversationMessage, AgentModelTurnRequest, AgentToolSpec
from vtm.agents.prompts import build_system_prompt


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

        def __exit__(self, _exc_type, _exc, _tb) -> None:
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


def test_openai_agent_adapter_coerces_single_command_payload() -> None:
    class FakeCommandClient:
        def create_chat_completion(self, **kwargs: object) -> dict[str, object]:
            del kwargs
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "command": "read",
                                    "arguments": {"path": "bug.py"},
                                }
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

    adapter = OpenAICompatibleAgentModelAdapter(model="gpt-test", client=FakeCommandClient())
    request = AgentModelTurnRequest(
        mode="benchmark_autonomous",
        prompt_profile="vtm-native-agent-v1",
        workspace="/tmp/workspace",
        task_payload={},
        messages=(AgentConversationMessage(role="user", content="Fix it"),),
        tools=(AgentToolSpec(name="read", description="Read a file"),),
    )

    response = adapter.complete_turn(request)

    assert response.done is False
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].tool_name == "read"
    assert response.tool_calls[0].arguments == {"path": "bug.py"}


def test_openai_agent_adapter_coerces_tool_calls_command_alias() -> None:
    class FakeToolCallsClient:
        def create_chat_completion(self, **kwargs: object) -> dict[str, object]:
            del kwargs
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "tool_calls": [
                                        {
                                            "command": "read",
                                            "arguments": {"path": "bug.py"},
                                        }
                                    ]
                                }
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

    adapter = OpenAICompatibleAgentModelAdapter(model="gpt-test", client=FakeToolCallsClient())
    request = AgentModelTurnRequest(
        mode="benchmark_autonomous",
        prompt_profile="vtm-native-agent-v1",
        workspace="/tmp/workspace",
        task_payload={},
        messages=(AgentConversationMessage(role="user", content="Fix it"),),
        tools=(AgentToolSpec(name="read", description="Read a file"),),
    )

    response = adapter.complete_turn(request)

    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].tool_name == "read"
    assert response.tool_calls[0].arguments == {"path": "bug.py"}


def test_openai_agent_adapter_strips_extra_tool_call_fields() -> None:
    class FakeExtraFieldsClient:
        def create_chat_completion(self, **kwargs: object) -> dict[str, object]:
            del kwargs
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "tool_calls": [
                                        {
                                            "tool_name": "read",
                                            "arguments": {"path": "bug.py"},
                                            "tool_call_id": "call_1",
                                            "type": "function",
                                        }
                                    ]
                                }
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

    adapter = OpenAICompatibleAgentModelAdapter(model="gpt-test", client=FakeExtraFieldsClient())
    request = AgentModelTurnRequest(
        mode="benchmark_autonomous",
        prompt_profile="vtm-native-agent-v1",
        workspace="/tmp/workspace",
        task_payload={},
        messages=(AgentConversationMessage(role="user", content="Fix it"),),
        tools=(AgentToolSpec(name="read", description="Read a file"),),
    )

    response = adapter.complete_turn(request)

    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].tool_name == "read"
    assert response.tool_calls[0].arguments == {"path": "bug.py"}


def test_openai_agent_adapter_accepts_tool_args_alias() -> None:
    class FakeToolArgsClient:
        def create_chat_completion(self, **kwargs: object) -> dict[str, object]:
            del kwargs
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "tool_calls": [
                                        {
                                            "tool_name": "read",
                                            "tool_args": {"path": "bug.py"},
                                        }
                                    ]
                                }
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

    adapter = OpenAICompatibleAgentModelAdapter(model="gpt-test", client=FakeToolArgsClient())
    request = AgentModelTurnRequest(
        mode="benchmark_autonomous",
        prompt_profile="vtm-native-agent-v1",
        workspace="/tmp/workspace",
        task_payload={},
        messages=(AgentConversationMessage(role="user", content="Fix it"),),
        tools=(AgentToolSpec(name="read", description="Read a file"),),
    )

    response = adapter.complete_turn(request)

    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].tool_name == "read"
    assert response.tool_calls[0].arguments == {"path": "bug.py"}


def test_build_system_prompt_includes_memory_and_anti_loop_guidance() -> None:
    request = AgentModelTurnRequest(
        mode="benchmark_autonomous",
        prompt_profile="vtm-native-agent-rlm-v1",
        workspace="/tmp/workspace",
        task_payload={
            "memory_mode": "lexical_rlm_rerank",
            "memory_context": [{"title": "rst writer", "summary": "related memory"}],
            "failing_tests": ["tests/test_rst.py::test_rst_with_header_rows"],
            "expected_changed_paths": ["astropy/io/ascii/rst.py"],
            "problem_statement": "Support header rows in RestructuredText output",
        },
        messages=(AgentConversationMessage(role="user", content="Fix it"),),
        tools=(
            AgentToolSpec(name="read", description="Read a file"),
            AgentToolSpec(name="retrieve_memory", description="Retrieve memory"),
            AgentToolSpec(name="apply_patch", description="Apply a patch"),
            AgentToolSpec(name="terminal", description="Run terminal commands"),
        ),
    )

    prompt = build_system_prompt(request)

    assert "Anti-loop rules:" in prompt
    assert "Do not repeat the same read/search call" in prompt
    assert "memory_mode=lexical_rlm_rerank" in prompt
    assert "Retrieved memory may already be model-reranked" in prompt
    assert "Expected changed paths" in prompt
    assert "- retrieve_memory:" in prompt


def test_openai_agent_adapter_forwards_built_prompt_to_system_message() -> None:
    client = FakeChatClient()
    adapter = OpenAICompatibleAgentModelAdapter(model="gpt-test", client=client)
    request = AgentModelTurnRequest(
        mode="benchmark_autonomous",
        prompt_profile="vtm-native-agent-v1",
        workspace="/tmp/workspace",
        task_payload={
            "memory_mode": "lexical",
            "memory_context": [{"title": "writer", "summary": "context"}],
            "expected_changed_paths": ["bug.py"],
        },
        messages=(AgentConversationMessage(role="user", content="Fix it"),),
        tools=(AgentToolSpec(name="read", description="Read a file"),),
    )

    adapter.complete_turn(request)

    assert client.last_kwargs is not None
    messages = client.last_kwargs["messages"]
    assert isinstance(messages, list)
    system_message = messages[0]
    assert isinstance(system_message, dict)
    content = system_message["content"]
    assert isinstance(content, str)
    assert "Task workflow:" in content
    assert "Anti-loop rules:" in content
    assert "memory_context retrieved by VTM" in content
