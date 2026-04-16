"""Built-in file and patch tools for the native agent runtime."""

from __future__ import annotations

from pathlib import Path

from vtm.agents.models import AgentToolResult, AgentToolSpec
from vtm.agents.tool_base import AgentTool, ToolExecutionContext
from vtm.agents.tool_utils import write_text_artifact


def build_read_tool() -> AgentTool:
    """Build the workspace file-read tool."""
    return AgentTool(
        spec=AgentToolSpec(
            name="read",
            description="Read a file inside the workspace.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer"},
                    "end_line": {"type": "integer"},
                    "max_chars": {"type": "integer"},
                },
                "required": ["path"],
            },
        ),
        handler=_read,
    )


def build_search_tool() -> AgentTool:
    """Build the workspace text-search tool."""
    return AgentTool(
        spec=AgentToolSpec(
            name="search",
            description="Search workspace text with ripgrep.",
            input_schema={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string"},
                },
                "required": ["pattern"],
            },
        ),
        handler=_search,
    )


def build_apply_patch_tool() -> AgentTool:
    """Build the git-apply patch tool."""
    return AgentTool(
        spec=AgentToolSpec(
            name="apply_patch",
            description="Apply a unified diff patch in the workspace.",
            input_schema={
                "type": "object",
                "properties": {"patch": {"type": "string"}},
                "required": ["patch"],
            },
        ),
        handler=_apply_patch,
    )


def _read(
    arguments: dict[str, object],
    context: ToolExecutionContext,
    call_id: str,
) -> AgentToolResult:
    """Read a workspace file slice and persist the rendered output as an artifact."""
    raw_path = str(arguments.get("path", ""))
    start_line = _coerce_required_int(arguments.get("start_line"), default=1)
    end_line = _coerce_optional_int(arguments.get("end_line"))
    max_chars = _coerce_required_int(arguments.get("max_chars"), default=20000)
    try:
        rendered = context.workspace_driver.read_file(
            raw_path,
            start_line=start_line,
            end_line=end_line,
            max_chars=max_chars,
        )
    except (FileNotFoundError, ValueError):
        path_name = Path(raw_path).name or raw_path
        return AgentToolResult(success=False, content=f"file not found: {path_name}")
    artifact_path, artifact_id = write_text_artifact(
        context=context,
        call_id=call_id,
        suffix=".read.txt",
        text=rendered,
        content_type="text/plain",
        metadata={"path": raw_path},
    )
    return AgentToolResult(
        success=True,
        content=rendered,
        metadata={"path": raw_path},
        artifact_path=artifact_path,
        artifact_id=artifact_id,
    )


def _coerce_optional_int(value: object) -> int | None:
    """Coerce an optional tool argument into an integer."""
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float, str)):
        return int(value)
    raise ValueError("expected an int-compatible value")


def _coerce_required_int(value: object, *, default: int) -> int:
    """Coerce a required-or-default tool argument into an integer."""
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float, str)):
        return int(value)
    raise ValueError("expected an int-compatible value")


def _search(
    arguments: dict[str, object],
    context: ToolExecutionContext,
    call_id: str,
) -> AgentToolResult:
    """Run workspace text search and persist the raw search output."""
    pattern = str(arguments.get("pattern", "")).strip()
    if not pattern:
        return AgentToolResult(success=False, content="search pattern must be non-empty")
    target = str(arguments.get("path", "."))
    completed = context.workspace_driver.search(pattern, path=target)
    artifact_path, artifact_id = write_text_artifact(
        context=context,
        call_id=call_id,
        suffix=".search.txt",
        text=completed.output,
        content_type="text/plain",
        metadata={
            "pattern": pattern,
            "command": completed.command,
            "duration_ms": completed.duration_ms,
            "exit_code": completed.exit_code,
            "timed_out": completed.timed_out,
            "truncated": completed.truncated,
        },
    )
    return AgentToolResult(
        success=(completed.exit_code in {0, 1}) and not completed.timed_out,
        content=completed.output,
        metadata={
            "pattern": pattern,
            "command": completed.command,
            "duration_ms": completed.duration_ms,
            "exit_code": completed.exit_code,
            "timed_out": completed.timed_out,
            "truncated": completed.truncated,
        },
        artifact_path=artifact_path,
        artifact_id=artifact_id,
    )


def _apply_patch(
    arguments: dict[str, object],
    context: ToolExecutionContext,
    call_id: str,
) -> AgentToolResult:
    """Apply a unified patch inside the workspace and record the command result."""
    patch_text = str(arguments.get("patch", ""))
    if not patch_text.strip():
        return AgentToolResult(success=False, content="patch must be non-empty")
    patch_path = context.artifact_root / "tool-results" / f"{call_id}.patch"
    applied = context.workspace_driver.apply_patch(patch_text, patch_path=patch_path)
    if applied.exit_code != 0 or applied.timed_out:
        failure_path, artifact_id = write_text_artifact(
            context=context,
            call_id=call_id,
            suffix=".apply.stderr.txt",
            text=applied.output,
            content_type="text/plain",
            metadata={
                "command": applied.command,
                "duration_ms": applied.duration_ms,
                "exit_code": applied.exit_code,
                "timed_out": applied.timed_out,
                "truncated": applied.truncated,
            },
        )
        return AgentToolResult(
            success=False,
            content=applied.output,
            metadata={
                "command": applied.command,
                "duration_ms": applied.duration_ms,
                "exit_code": applied.exit_code,
                "timed_out": applied.timed_out,
                "truncated": applied.truncated,
                "patch_path": str(patch_path),
            },
            artifact_path=failure_path,
            artifact_id=artifact_id,
        )
    artifact_path, artifact_id = write_text_artifact(
        context=context,
        call_id=call_id,
        suffix=".apply.txt",
        text=applied.output or "patch applied",
        content_type="text/plain",
        metadata={
            "command": applied.command,
            "duration_ms": applied.duration_ms,
            "exit_code": applied.exit_code,
            "timed_out": applied.timed_out,
            "truncated": applied.truncated,
        },
    )
    return AgentToolResult(
        success=True,
        content=applied.output or "patch applied",
        metadata={
            "command": applied.command,
            "duration_ms": applied.duration_ms,
            "exit_code": applied.exit_code,
            "timed_out": applied.timed_out,
            "truncated": applied.truncated,
            "patch_path": str(patch_path),
        },
        artifact_path=artifact_path,
        artifact_id=artifact_id,
    )


__all__ = [
    "build_apply_patch_tool",
    "build_read_tool",
    "build_search_tool",
]
