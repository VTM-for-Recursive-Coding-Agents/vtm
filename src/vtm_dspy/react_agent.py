"""Minimal DSPy ReAct wrapper for controlled VTM coding workflows."""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from vtm.fingerprints import DependencyFingerprint
from vtm.memory_items import MemoryItem, VisibilityScope
from vtm.services.memory_kernel import MemoryKernel

from . import require_dspy
from .config import DSPyOpenRouterConfig
from .tools import VTMMemoryTools, WorkspaceTools

DEFAULT_REACT_MAX_ITERS = 20


def _sequence_length(value: Any) -> int | None:
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return len(value)
    return None


def _usage_value(payload: Mapping[str, Any] | None, *keys: str) -> int | None:
    if payload is None:
        return None
    for key in keys:
        value = payload.get(key)
        if isinstance(value, int):
            return value
    return None


def _normalize_usage(payload: Any) -> dict[str, Any] | None:
    if isinstance(payload, Mapping):
        return {str(key): value for key, value in payload.items()}
    return None


def _extract_finish_reasons(response: Any) -> list[str]:
    choices = None
    if isinstance(response, Mapping):
        choices = response.get("choices")
    else:
        choices = getattr(response, "choices", None)
    if not isinstance(choices, Sequence) or isinstance(choices, str | bytes):
        return []
    finish_reasons: list[str] = []
    for choice in choices:
        finish_reason = (
            choice.get("finish_reason")
            if isinstance(choice, Mapping)
            else getattr(choice, "finish_reason", None)
        )
        if isinstance(finish_reason, str) and finish_reason.strip():
            finish_reasons.append(finish_reason)
    return finish_reasons


def _count_output_tool_calls(outputs: Any) -> int:
    if isinstance(outputs, Mapping):
        tool_calls = outputs.get("tool_calls")
        if isinstance(tool_calls, Sequence) and not isinstance(tool_calls, str | bytes):
            return len(tool_calls)
        return 0
    if isinstance(outputs, Sequence) and not isinstance(outputs, str | bytes):
        return sum(_count_output_tool_calls(item) for item in outputs)
    return 0


def _tool_name(instance: Any) -> str:
    name = getattr(instance, "name", None)
    if isinstance(name, str) and name.strip():
        return name
    return type(instance).__name__


class _DSPyTraceRecorder:
    """Capture lightweight LM and tool timing for one DSPy run."""

    def __init__(self) -> None:
        self._active_lm: dict[str, dict[str, Any]] = {}
        self._active_tools: dict[str, dict[str, Any]] = {}
        self.lm_calls: list[dict[str, Any]] = []
        self.tool_calls: list[dict[str, Any]] = []

    def on_lm_start(self, call_id: str, instance: Any, inputs: dict[str, Any]) -> None:
        history = getattr(instance, "history", None)
        self._active_lm[call_id] = {
            "started_at": time.perf_counter(),
            "instance": instance,
            "history_len": len(history) if isinstance(history, Sequence) else 0,
            "message_count": _sequence_length(inputs.get("messages")),
            "prompt_chars": (
                len(inputs["prompt"]) if isinstance(inputs.get("prompt"), str) else None
            ),
            "requested_max_tokens": inputs.get("kwargs", {}).get("max_tokens")
            if isinstance(inputs.get("kwargs"), Mapping)
            else None,
        }

    def on_lm_end(
        self,
        call_id: str,
        outputs: dict[str, Any] | None,
        exception: Exception | None = None,
    ) -> None:
        active = self._active_lm.pop(call_id, {})
        instance = active.get("instance")
        duration_ms = round(
            (time.perf_counter() - active.get("started_at", time.perf_counter())) * 1000,
            3,
        )
        history_entry = self._latest_history_entry(instance, active.get("history_len", 0))
        usage = _normalize_usage(history_entry.get("usage") if history_entry is not None else None)
        finish_reasons = _extract_finish_reasons(
            history_entry.get("response") if history_entry is not None else None
        )
        prompt_tokens = _usage_value(usage, "prompt_tokens", "input_tokens")
        completion_tokens = _usage_value(usage, "completion_tokens", "output_tokens")
        self.lm_calls.append(
            {
                "duration_ms": duration_ms,
                "model": history_entry.get("model") if history_entry is not None else None,
                "response_model": history_entry.get("response_model")
                if history_entry is not None
                else None,
                "message_count": active.get("message_count"),
                "prompt_chars": active.get("prompt_chars"),
                "requested_max_tokens": active.get("requested_max_tokens"),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "finish_reasons": finish_reasons,
                "truncated": "length" in finish_reasons,
                "output_tool_calls": _count_output_tool_calls(outputs),
                "exception": str(exception) if exception is not None else None,
            }
        )

    def on_tool_start(self, call_id: str, instance: Any, inputs: dict[str, Any]) -> None:
        self._active_tools[call_id] = {
            "started_at": time.perf_counter(),
            "tool_name": _tool_name(instance),
            "input_keys": sorted(inputs.keys()),
        }

    def on_tool_end(
        self,
        call_id: str,
        outputs: dict[str, Any] | None,
        exception: Exception | None = None,
    ) -> None:
        active = self._active_tools.pop(call_id, {})
        self.tool_calls.append(
            {
                "tool_name": active.get("tool_name"),
                "duration_ms": round(
                    (time.perf_counter() - active.get("started_at", time.perf_counter())) * 1000,
                    3,
                ),
                "input_keys": active.get("input_keys"),
                "output_type": type(outputs).__name__ if outputs is not None else None,
                "exception": str(exception) if exception is not None else None,
            }
        )

    def summary(self) -> dict[str, Any]:
        finish_reason_counts = Counter(
            reason
            for call in self.lm_calls
            for reason in call.get("finish_reasons", [])
            if isinstance(reason, str)
        )
        total_prompt_tokens = sum(
            value
            for call in self.lm_calls
            for value in [call.get("prompt_tokens")]
            if isinstance(value, int)
        )
        total_completion_tokens = sum(
            value
            for call in self.lm_calls
            for value in [call.get("completion_tokens")]
            if isinstance(value, int)
        )
        return {
            "lm_call_count": len(self.lm_calls),
            "tool_call_count": len(self.tool_calls),
            "total_lm_duration_ms": round(
                sum(call["duration_ms"] for call in self.lm_calls),
                3,
            ),
            "max_lm_duration_ms": round(
                max((call["duration_ms"] for call in self.lm_calls), default=0.0),
                3,
            ),
            "truncated_lm_call_count": sum(1 for call in self.lm_calls if call["truncated"]),
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "finish_reason_counts": dict(sorted(finish_reason_counts.items())),
            "lm_calls": self.lm_calls,
            "tool_calls": self.tool_calls,
        }

    def _latest_history_entry(
        self,
        instance: Any,
        history_len: int,
    ) -> Mapping[str, Any] | None:
        history = getattr(instance, "history", None)
        if not isinstance(history, Sequence) or isinstance(history, str | bytes):
            return None
        if len(history) <= history_len:
            return None
        candidate = history[-1]
        if isinstance(candidate, Mapping):
            return candidate
        return None


class VTMReActCodingAgent:
    """Optional DSPy ReAct wrapper that keeps VTM as the memory kernel."""

    def __init__(
        self,
        *,
        kernel: MemoryKernel | None,
        scopes: Sequence[VisibilityScope] = (),
        enable_memory_tools: bool = True,
        enable_memory_write_tools: bool | None = None,
        workspace_root: str | Path | None = None,
        dependency_provider: Callable[[], DependencyFingerprint | None] | None = None,
        memory_lookup: Callable[[str], MemoryItem | None] | None = None,
        model_config: DSPyOpenRouterConfig | None = None,
        command_timeout_seconds: int = 120,
        max_output_chars: int = 20000,
        max_iters: int = DEFAULT_REACT_MAX_ITERS,
    ) -> None:
        self.model_config = model_config or DSPyOpenRouterConfig.from_env()
        self.max_iters = max(1, int(max_iters))
        resolved_enable_memory_write_tools = (
            enable_memory_tools
            if enable_memory_write_tools is None
            else bool(enable_memory_write_tools)
        )
        self.memory_tools = VTMMemoryTools(
            kernel=kernel,
            scopes=scopes,
            dependency_provider=dependency_provider,
            memory_lookup=memory_lookup,
            enable_lookup_tools=enable_memory_tools,
            enable_write_tools=resolved_enable_memory_write_tools,
        )
        self.workspace_tools = (
            WorkspaceTools(
                workspace_root,
                command_timeout_seconds=command_timeout_seconds,
                max_output_chars=max_output_chars,
            )
            if workspace_root is not None
            else None
        )

    def tool_mapping(self) -> dict[str, Callable[..., Any]]:
        """Return the full tool set exposed to a DSPy ReAct program."""
        mapping = dict(self.memory_tools.tool_mapping())
        if self.workspace_tools is not None:
            mapping.update(self.workspace_tools.tool_mapping())
        return mapping

    def tool_names(self) -> tuple[str, ...]:
        """Return the stable ordered tool names for dry-run inspection."""
        return tuple(self.tool_mapping().keys())

    def describe(self) -> dict[str, Any]:
        """Return dry-run metadata describing the configured agent surface."""
        return {
            "tool_names": list(self.tool_names()),
            "workspace_root": (
                str(self.workspace_tools.workspace_root)
                if self.workspace_tools is not None
                else None
            ),
            "model": self.model_config.summary(),
            "memory_tools_enabled": self.memory_tools.enabled,
            "workspace_tools_enabled": self.workspace_tools is not None,
            "react_max_iters": self.max_iters if self.tool_mapping() else None,
        }

    def create_lm(self, *, callbacks: Sequence[Any] = ()) -> Any:
        """Instantiate the configured DSPy LM using OpenRouter-compatible settings."""
        dspy = require_dspy()
        lm_kwargs = dict(self.model_config.lm_kwargs())
        lm_kwargs.setdefault("api_key", self.model_config.require_api_key())
        if callbacks:
            lm_kwargs["callbacks"] = list(callbacks)
        return dspy.LM(self.model_config.lm_model_name(), **lm_kwargs)

    def create_program(
        self,
        *,
        signature: str = "task -> response",
        force_plain: bool = False,
        callbacks: Sequence[Any] = (),
    ) -> Any:
        """Construct a DSPy program suitable for the configured tool surface."""
        dspy = require_dspy()
        lm = self.create_lm(callbacks=callbacks)
        tools = list(self.tool_mapping().values())
        program = (
            dspy.Predict(signature)
            if force_plain or not tools
            else dspy.ReAct(signature, tools=tools, max_iters=self.max_iters)
        )
        if hasattr(program, "set_lm"):
            program.set_lm(lm)
            return program
        if hasattr(dspy, "configure"):
            dspy.configure(lm=lm)
        return program

    def run(self, task: str, *, signature: str = "task -> response") -> dict[str, Any]:
        """Execute one DSPy ReAct trajectory and capture the resulting patch metadata."""
        if not task.strip():
            raise ValueError("task must be non-empty")
        self.memory_tools.clear_write_proposals()
        uses_tools = bool(self.tool_mapping())
        execution_mode = "react" if uses_tools else "predict"
        fallback_error: str | None = None
        trace_recorder = _DSPyTraceRecorder()
        program = self.create_program(signature=signature, callbacks=(trace_recorder,))
        try:
            prediction = program(task=task)
        except Exception as exc:
            if not uses_tools or not self._looks_like_react_schema_failure(exc):
                raise
            program = self.create_program(
                signature=signature,
                force_plain=True,
                callbacks=(trace_recorder,),
            )
            prediction = program(task=task)
            execution_mode = "predict_fallback"
            fallback_error = str(exc)
        diff_payload = self.workspace_tools.git_diff() if self.workspace_tools is not None else None
        memory_write_proposals = self.memory_tools.drain_write_proposals()
        return {
            "response": self._serialize_prediction(prediction),
            "patch": diff_payload["diff"] if diff_payload is not None else "",
            "memory_write_proposals": memory_write_proposals,
            "trajectory": {
                **self.describe(),
                "task_chars": len(task),
                "execution_mode": execution_mode,
                "fallback_error": fallback_error,
                "memory_write_proposal_count": len(memory_write_proposals),
                "diagnostics": trace_recorder.summary(),
            },
        }

    def _serialize_prediction(self, prediction: Any) -> Any:
        if hasattr(prediction, "model_dump"):
            return prediction.model_dump()
        if hasattr(prediction, "toDict"):
            return prediction.toDict()
        if hasattr(prediction, "__dict__"):
            return dict(prediction.__dict__)
        return prediction

    def _looks_like_react_schema_failure(self, exc: Exception) -> bool:
        message = str(exc)
        return "next_tool_name" in message or "Adapter" in exc.__class__.__name__


__all__ = ["VTMReActCodingAgent"]
