"""Prompt builders for native-agent model turns."""

from __future__ import annotations

import json
from typing import Any

from vtm.agents.models import AgentModelTurnRequest


def build_system_prompt(request: AgentModelTurnRequest) -> str:
    """Build the system prompt for the configured native-agent profile."""
    profile = request.prompt_profile.strip() or "vtm-native-agent-v1"
    lines = [
        "You are VTM's single-agent coding runtime operating inside a local repository workspace.",
        "Return only a JSON object matching the requested schema.",
        "Use only declared tools. Never invent tool outputs, file contents, test results, or patches.",
        "If you are not finished, return tool_calls and set done=false.",
        "If the task is complete or you are blocked after taking reasonable actions, set done=true.",
        "Keep tool arguments minimal and valid for the declared tool schemas.",
        "",
        f"Prompt profile: {profile}",
        f"Workspace: {request.workspace}",
        f"Mode: {request.mode.value}",
        f"Max tool calls this turn: {request.max_tool_calls}",
        "",
        "Output contract:",
        '- assistant_message: optional short status or completion summary',
        '- tool_calls: array of {"tool_name": string, "arguments": object}',
        "- done: boolean",
        "",
        "Task workflow:",
        "1. Start from the task payload. Prefer touched_paths, expected_changed_paths, failing_tests, and memory_context over broad exploration.",
        "2. Use search to locate symbols or call sites, then read targeted files or line ranges.",
        "3. Once the bug and fix are clear, use apply_patch to make the smallest plausible change.",
        "4. Use terminal for targeted verification after patching or when command output is needed.",
        "5. Finish only after patching and attempting verification, or after a concise blocked explanation.",
        "",
        "Anti-loop rules:",
        "1. Do not repeat the same read/search call on the same target unless you changed the file, changed the line range, or learned something new that justifies it.",
        "2. If you have already read the primary file, switch to search, terminal, apply_patch, or another relevant file instead of rereading it unchanged.",
        "3. If two consecutive turns made no progress, take a different action.",
        "",
        "Tool guidance:",
        "- read: inspect a specific file or slice when you know where to look.",
        "- search: locate symbols, tests, constructor signatures, or related paths before reading.",
        "- apply_patch: write the actual fix as a unified diff once the change is scoped.",
        "- terminal: run focused commands such as rg, sed, python, pytest, or git diff/status for evidence and verification.",
        "- retrieve_memory: query VTM memory when repository context or prior task learnings may help.",
        "- record_task_memory: save only verified, reusable facts discovered during this run.",
        "- promote_procedure: use only after a successful, reusable repair pattern is established.",
    ]
    lines.extend(_memory_guidance(request))
    lines.extend(_verification_guidance(request.task_payload))
    lines.extend(_task_context_guidance(request.task_payload))
    lines.extend(_tool_catalog(request))
    return "\n".join(lines)


def _memory_guidance(request: AgentModelTurnRequest) -> list[str]:
    payload = request.task_payload
    memory_context = payload.get("memory_context")
    memory_mode = str(payload.get("memory_mode", "")).strip() or "unknown"
    lines = [
        "",
        "Memory guidance:",
        f"- The task payload may already contain seeded memory_context retrieved by VTM (memory_mode={memory_mode}). Use it as a starting hint set, not as ground truth.",
        "- If memory_context looks insufficient or overly broad, use retrieve_memory with a sharper query tied to the failing behavior, symbol, or file.",
    ]
    if memory_mode == "lexical_rlm_rerank" or request.prompt_profile == "vtm-native-agent-rlm-v1":
        lines.append(
            "- Retrieved memory may already be model-reranked. Prefer high-specificity memories, but still verify against workspace files before acting."
        )
    if isinstance(memory_context, list) and memory_context:
        lines.append(
            f"- The current task payload already includes {len(memory_context)} memory candidates; inspect them before issuing broad repository exploration."
        )
    return lines


def _verification_guidance(task_payload: dict[str, Any]) -> list[str]:
    failing_tests = _string_tuple(task_payload.get("failing_tests"))
    fail_to_pass = _string_tuple(task_payload.get("fail_to_pass_tests"))
    pass_to_pass = _string_tuple(task_payload.get("pass_to_pass_tests"))
    lines = [
        "",
        "Verification guidance:",
    ]
    if fail_to_pass:
        lines.append(
            f"- Prioritize the fail_to_pass tests first: {json.dumps(fail_to_pass[:5])}"
        )
    elif failing_tests:
        lines.append(f"- Prioritize the failing tests: {json.dumps(failing_tests[:5])}")
    if pass_to_pass:
        lines.append(
            "- After the fix, spot-check pass_to_pass coverage when practical to avoid regressions."
        )
    lines.append(
        "- If the task payload provides test_command, prefer that exact command for final verification."
    )
    return lines


def _task_context_guidance(task_payload: dict[str, Any]) -> list[str]:
    touched_paths = _string_tuple(task_payload.get("touched_paths"))
    expected_changed_paths = _string_tuple(task_payload.get("expected_changed_paths"))
    problem_statement = str(task_payload.get("problem_statement") or "").strip()
    lines = [
        "",
        "Task context:",
    ]
    if expected_changed_paths:
        lines.append(
            f"- Expected changed paths: {json.dumps(expected_changed_paths[:10])}"
        )
    if touched_paths:
        lines.append(f"- Candidate touched paths: {json.dumps(touched_paths[:10])}")
    if problem_statement:
        lines.append(
            f"- Problem statement excerpt: {json.dumps(problem_statement[:500])}"
        )
    return lines


def _tool_catalog(request: AgentModelTurnRequest) -> list[str]:
    lines = [
        "",
        "Declared tools:",
    ]
    for tool in request.tools:
        required = tool.input_schema.get("required", [])
        required_text = ", ".join(str(item) for item in required) if required else "none"
        lines.append(
            f"- {tool.name}: {tool.description} Required args: {required_text}."
        )
    return lines


def _string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(str(item) for item in value)


__all__ = ["build_system_prompt"]
