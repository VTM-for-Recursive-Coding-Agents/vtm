"""Shared helpers for built-in agent tools."""

from __future__ import annotations

from typing import Any

from vtm.agents.tool_base import ToolExecutionContext
from vtm.enums import EvidenceKind
from vtm.memory_items import MemoryItem


def write_text_artifact(
    *,
    context: ToolExecutionContext,
    call_id: str,
    suffix: str,
    text: str,
    content_type: str,
    metadata: dict[str, Any],
) -> tuple[str, str | None]:
    """Write a text artifact locally and optionally capture it in the kernel."""
    tool_results_root = context.artifact_root / "tool-results"
    tool_results_root.mkdir(parents=True, exist_ok=True)
    path = tool_results_root / f"{call_id}{suffix}"
    path.write_text(text, encoding="utf-8")
    artifact_id: str | None = None
    if context.kernel is not None:
        artifact = context.kernel.capture_artifact(
            text.encode("utf-8"),
            content_type=content_type,
            tool_name=context.tool_name_prefix,
            metadata=metadata,
        )
        artifact_id = artifact.artifact_id
    return str(path), artifact_id


def write_local_text_artifact(
    *,
    context: ToolExecutionContext,
    call_id: str,
    suffix: str,
    text: str,
) -> str:
    """Write a text artifact locally without creating a kernel artifact."""
    tool_results_root = context.artifact_root / "tool-results"
    tool_results_root.mkdir(parents=True, exist_ok=True)
    path = tool_results_root / f"{call_id}{suffix}"
    path.write_text(text, encoding="utf-8")
    return str(path)


def first_anchor_path(memory: MemoryItem) -> str | None:
    """Return the first code-anchor path attached to a memory item."""
    for evidence in memory.evidence:
        if evidence.kind is EvidenceKind.CODE_ANCHOR and evidence.code_anchor is not None:
            return evidence.code_anchor.path
    return None


__all__ = [
    "first_anchor_path",
    "write_local_text_artifact",
    "write_text_artifact",
]
