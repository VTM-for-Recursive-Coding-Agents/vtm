# src/vtm/agents

Purpose: native single-agent runtime package.

Start here
- `runtime.py`: the `TerminalCodingAgent` loop and runtime context.
- `models.py`: all durable request, trace, and result records.
- `permissions.py` and `tools.py`: execution policy and built-in tool registry.

Contents
- `runtime.py`: `TerminalCodingAgent` loop and runtime context.
- `models.py`: Agent request/result, prompt, turn, tool-call, and compaction records.
- `permissions.py`: Guarded and benchmark-autonomous tool policies.
- `tools.py`: Public tool provider entrypoint.
- `tool_*.py`: Terminal, file, memory, and shared tool helpers.
- `workspace.py`: Compatibility shim re-exporting the public harness workspace surface.
