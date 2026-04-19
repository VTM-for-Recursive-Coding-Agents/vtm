"""Minimal DSPy ReAct wrapper for controlled VTM coding workflows."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from vtm.fingerprints import DependencyFingerprint
from vtm.memory_items import MemoryItem, VisibilityScope
from vtm.services.memory_kernel import MemoryKernel

from . import require_dspy
from .config import DSPyOpenRouterConfig
from .tools import VTMMemoryTools, WorkspaceTools


class VTMReActCodingAgent:
    """Optional DSPy ReAct wrapper that keeps VTM as the memory kernel."""

    def __init__(
        self,
        *,
        kernel: MemoryKernel | None,
        scopes: Sequence[VisibilityScope] = (),
        workspace_root: str | Path | None = None,
        dependency_provider: Callable[[], DependencyFingerprint | None] | None = None,
        memory_lookup: Callable[[str], MemoryItem | None] | None = None,
        model_config: DSPyOpenRouterConfig | None = None,
        command_timeout_seconds: int = 120,
        max_output_chars: int = 20000,
    ) -> None:
        self.model_config = model_config or DSPyOpenRouterConfig.from_env()
        self.memory_tools = VTMMemoryTools(
            kernel=kernel,
            scopes=scopes,
            dependency_provider=dependency_provider,
            memory_lookup=memory_lookup,
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
        }

    def create_lm(self) -> Any:
        """Instantiate the configured DSPy LM using OpenRouter-compatible settings."""
        dspy = require_dspy()
        lm_kwargs = dict(self.model_config.lm_kwargs())
        lm_kwargs.setdefault("api_key", self.model_config.require_api_key())
        return dspy.LM(self.model_config.lm_model_name(), **lm_kwargs)

    def create_program(self, *, signature: str = "task -> response") -> Any:
        """Construct a minimal DSPy ReAct agent without executing it."""
        dspy = require_dspy()
        lm = self.create_lm()
        react = dspy.ReAct(signature, tools=list(self.tool_mapping().values()))
        if hasattr(react, "set_lm"):
            react.set_lm(lm)
            return react
        if hasattr(dspy, "configure"):
            dspy.configure(lm=lm)
        return react

    def run(self, task: str, *, signature: str = "task -> response") -> dict[str, Any]:
        """Execute one DSPy ReAct trajectory and capture the resulting patch metadata."""
        if not task.strip():
            raise ValueError("task must be non-empty")
        program = self.create_program(signature=signature)
        prediction = program(task=task)
        diff_payload = self.workspace_tools.git_diff() if self.workspace_tools is not None else None
        return {
            "response": self._serialize_prediction(prediction),
            "patch": diff_payload["diff"] if diff_payload is not None else "",
            "trajectory": {
                **self.describe(),
                "task": task,
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


__all__ = ["VTMReActCodingAgent"]
