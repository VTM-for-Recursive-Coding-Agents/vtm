"""Shared tool abstractions for the native agent runtime."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from vtm.agents.models import AgentToolResult, AgentToolSpec
from vtm.harness.workspace import WorkspaceDriver
from vtm.memory_items import VisibilityScope
from vtm.services import DependencyFingerprintBuilder, TransactionalMemoryKernel


@dataclass(frozen=True)
class ToolExecutionContext:
    """Runtime context passed to built-in tool handlers."""

    workspace_root: Path
    task_file: Path
    task_payload: dict[str, Any]
    artifact_root: Path
    workspace_driver: WorkspaceDriver
    command_timeout_seconds: int = 120
    max_command_output_chars: int = 20000
    kernel: TransactionalMemoryKernel | None = None
    task_scope: VisibilityScope | None = None
    durable_scope: VisibilityScope | None = None
    dependency_builder: DependencyFingerprintBuilder | None = None
    tool_name_prefix: str = "agent-tool"


class AgentTool:
    """Executable tool wrapper exposed to the runtime."""

    def __init__(
        self,
        *,
        spec: AgentToolSpec,
        handler: Callable[[dict[str, Any], ToolExecutionContext, str], AgentToolResult],
    ) -> None:
        """Wrap a tool spec plus its execution handler."""
        self.spec = spec
        self._handler = handler

    def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
        call_id: str,
    ) -> AgentToolResult:
        """Execute the underlying handler with normalized arguments."""
        return self._handler(arguments, context, call_id)


class ToolProvider(Protocol):
    """Builds the tool registry for a runtime invocation."""

    def build_tools(self, context: ToolExecutionContext) -> Mapping[str, AgentTool]: ...


__all__ = [
    "AgentTool",
    "ToolExecutionContext",
    "ToolProvider",
]
