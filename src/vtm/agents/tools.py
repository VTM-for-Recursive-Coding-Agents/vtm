"""Built-in tool registry for the native agent runtime."""

from __future__ import annotations

from collections.abc import Mapping

from vtm.agents.tool_base import AgentTool, ToolExecutionContext, ToolProvider
from vtm.agents.tool_files import (
    build_apply_patch_tool,
    build_read_tool,
    build_search_tool,
)
from vtm.agents.tool_memory import (
    build_promote_procedure_tool,
    build_record_task_memory_tool,
    build_retrieve_memory_tool,
)
from vtm.agents.tool_terminal import build_terminal_tool


class BuiltInToolProvider:
    """Assembles the default built-in tool set for local coding tasks."""

    def build_tools(self, context: ToolExecutionContext) -> Mapping[str, AgentTool]:
        """Return the built-in tools available to the runtime."""
        del context
        return {
            "terminal": build_terminal_tool(),
            "read": build_read_tool(),
            "search": build_search_tool(),
            "apply_patch": build_apply_patch_tool(),
            "retrieve_memory": build_retrieve_memory_tool(),
            "record_task_memory": build_record_task_memory_tool(),
            "promote_procedure": build_promote_procedure_tool(),
        }


__all__ = [
    "AgentTool",
    "BuiltInToolProvider",
    "ToolExecutionContext",
    "ToolProvider",
]
