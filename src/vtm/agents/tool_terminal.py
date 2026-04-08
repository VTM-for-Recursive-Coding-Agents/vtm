"""Built-in terminal tool for the native agent runtime."""

from __future__ import annotations

from vtm.agents.models import AgentToolResult, AgentToolSpec
from vtm.agents.tool_base import AgentTool, ToolExecutionContext
from vtm.agents.tool_utils import write_text_artifact


def build_terminal_tool() -> AgentTool:
    """Build the persistent terminal-session tool."""
    return AgentTool(
        spec=AgentToolSpec(
            name="terminal",
            description="Run a terminal command in the persistent session.",
            input_schema={
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        ),
        handler=_terminal,
    )


def _terminal(
    arguments: dict[str, object],
    context: ToolExecutionContext,
    call_id: str,
) -> AgentToolResult:
    command = str(arguments.get("command", "")).strip()
    if not command:
        return AgentToolResult(success=False, content="terminal command must be non-empty")
    result = context.workspace_driver.run_terminal(
        command,
        timeout_seconds=context.command_timeout_seconds,
        max_output_chars=context.max_command_output_chars,
    )
    artifact_path, artifact_id = write_text_artifact(
        context=context,
        call_id=call_id,
        suffix=".terminal.txt",
        text=result.output,
        content_type="text/plain",
        metadata={
            "command": result.command,
            "duration_ms": result.duration_ms,
            "exit_code": result.exit_code,
            "timed_out": result.timed_out,
            "truncated": result.truncated,
        },
    )
    return AgentToolResult(
        success=result.exit_code == 0 and not result.timed_out,
        content=result.output,
        metadata={
            "command": result.command,
            "duration_ms": result.duration_ms,
            "exit_code": result.exit_code,
            "timed_out": result.timed_out,
            "truncated": result.truncated,
        },
        artifact_path=artifact_path,
        artifact_id=artifact_id,
    )


__all__ = ["build_terminal_tool"]
