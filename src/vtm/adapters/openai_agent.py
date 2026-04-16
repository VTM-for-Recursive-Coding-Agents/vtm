"""Optional OpenAI-compatible model adapter for the native agent runtime."""

from __future__ import annotations

import json
import os

from vtm.adapters.agent_model import AgentModelAdapter
from vtm.adapters.openai_chat import OpenAICompatibleChatClient, OpenAICompatibleChatConfig
from vtm.agents.prompts import build_system_prompt
from vtm.agents.models import AgentModelTurnRequest, AgentModelTurnResponse


class OpenAICompatibleAgentModelAdapter:
    """Drives the native agent using a chat-completions compatible endpoint."""

    def __init__(
        self,
        *,
        model: str,
        client: OpenAICompatibleChatClient | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        base_url_env: str = "VTM_AGENT_BASE_URL",
        api_key_env: str = "VTM_AGENT_API_KEY",
        timeout_seconds: int = 180,
        max_output_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> None:
        """Create an adapter around a model name and chat client."""
        if not model:
            raise ValueError("OpenAI-compatible agent adapter requires a non-empty model")
        self._model = model
        self._max_output_tokens = max_output_tokens
        self._temperature = temperature
        self._client = client or self._build_client(
            base_url=base_url,
            api_key=api_key,
            base_url_env=base_url_env,
            api_key_env=api_key_env,
            timeout_seconds=timeout_seconds,
        )

    @property
    def model_id(self) -> str:
        """Return the configured model identifier."""
        return self._model

    def complete_turn(self, request: AgentModelTurnRequest) -> AgentModelTurnResponse:
        """Execute one model turn and validate the returned JSON payload."""
        system_prompt = self._build_system_prompt(request)
        response_payload = self._client.create_chat_completion(
            model=self._model,
            temperature=request.sampling_temperature,
            max_tokens=self._max_output_tokens,
            seed=request.sampling_seed,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "mode": request.mode.value,
                            "tool_specs": [tool.model_dump(mode="json") for tool in request.tools],
                            "messages": [
                                message.model_dump(mode="json") for message in request.messages
                            ],
                            "task_payload": request.task_payload,
                            "workspace": request.workspace,
                            "max_tool_calls": request.max_tool_calls,
                        },
                        sort_keys=True,
                    ),
                },
            ],
        )
        text = self._client.extract_message_text(response_payload)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "OpenAI-compatible agent adapter did not return valid JSON"
            ) from exc
        if isinstance(payload, dict):
            payload = self._normalize_payload(payload)
        return AgentModelTurnResponse.model_validate(payload)

    def _build_client(
        self,
        *,
        base_url: str | None,
        api_key: str | None,
        base_url_env: str,
        api_key_env: str,
        timeout_seconds: int,
    ) -> OpenAICompatibleChatClient:
        resolved_base_url = base_url if base_url is not None else os.getenv(base_url_env, "")
        resolved_api_key = api_key if api_key is not None else os.getenv(api_key_env, "")
        resolved_base_url = resolved_base_url.strip()
        resolved_api_key = resolved_api_key.strip()
        if not resolved_base_url:
            raise ValueError(
                "OpenAI-compatible agent adapter requires base_url or the environment "
                f"variable {base_url_env!r}"
            )
        if not resolved_api_key:
            raise ValueError(
                "OpenAI-compatible agent adapter requires api_key or the environment "
                f"variable {api_key_env!r}"
            )
        return OpenAICompatibleChatClient(
            OpenAICompatibleChatConfig(
                base_url=resolved_base_url,
                api_key=resolved_api_key,
                timeout_seconds=timeout_seconds,
            )
        )

    def _build_system_prompt(self, request: AgentModelTurnRequest) -> str:
        return build_system_prompt(request)

    def _normalize_payload(self, payload: dict[str, object]) -> dict[str, object]:
        """Coerce common near-miss tool-call payloads into the expected schema."""
        if "tool_calls" in payload and isinstance(payload["tool_calls"], list):
            normalized_calls: list[dict[str, object]] = []
            changed = False
            for item in payload["tool_calls"]:
                if not isinstance(item, dict):
                    normalized_calls.append(item)
                    continue
                command = item.get("tool_name", item.get("command"))
                arguments = item.get("arguments", item.get("tool_args"))
                if isinstance(command, str) and isinstance(arguments, dict):
                    normalized_calls.append(
                        {
                            "tool_name": command,
                            "arguments": arguments,
                        }
                    )
                    changed = True
                    continue
                normalized_calls.append(item)
            if changed:
                return {**payload, "tool_calls": normalized_calls}

        command = payload.get("command")
        arguments = payload.get("arguments", payload.get("tool_args"))
        if isinstance(command, str) and isinstance(arguments, dict):
            assistant_message = payload.get("assistant_message")
            done = payload.get("done")
            metadata = payload.get("metadata")
            normalized: dict[str, object] = {
                "tool_calls": [{"tool_name": command, "arguments": arguments}],
            }
            if isinstance(assistant_message, str):
                normalized["assistant_message"] = assistant_message
            if isinstance(done, bool):
                normalized["done"] = done
            if isinstance(metadata, dict):
                normalized["metadata"] = metadata
            return normalized

        return payload


__all__ = ["AgentModelAdapter", "OpenAICompatibleAgentModelAdapter"]
