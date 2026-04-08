"""Permission policies that gate native-agent tool execution."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from vtm.base import VTMModel


class PermissionDecision(VTMModel):
    """Authorization result for one proposed tool call."""

    allowed: bool
    reason: str | None = None


class ToolPermissionPolicy(Protocol):
    """Policy interface used by the native runtime before executing tools."""

    def authorize(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        workspace_root: Path,
    ) -> PermissionDecision: ...


class BenchmarkAutonomousPermissionPolicy:
    """Permissive policy used for benchmark-controlled autonomous runs."""

    def authorize(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        workspace_root: Path,
    ) -> PermissionDecision:
        return PermissionDecision(allowed=True)


class InteractiveGuardedPermissionPolicy:
    """Simple local guardrail policy for interactive runs."""

    _blocked_fragments = (
        "rm -rf /",
        "git reset --hard",
        "git checkout --",
        "shutdown",
        "reboot",
    )

    def authorize(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        workspace_root: Path,
    ) -> PermissionDecision:
        if tool_name == "terminal":
            command = str(arguments.get("command", ""))
            lowered = command.lower()
            for fragment in self._blocked_fragments:
                if fragment in lowered:
                    return PermissionDecision(
                        allowed=False,
                        reason=f"blocked terminal fragment: {fragment}",
                    )
        return PermissionDecision(allowed=True)


__all__ = [
    "BenchmarkAutonomousPermissionPolicy",
    "InteractiveGuardedPermissionPolicy",
    "PermissionDecision",
    "ToolPermissionPolicy",
]
