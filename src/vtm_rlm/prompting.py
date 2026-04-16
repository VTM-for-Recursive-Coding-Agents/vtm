"""Prompt helpers for phase-1 VTM plus vendored-RLM execution."""

from __future__ import annotations

from pathlib import Path

from vtm.harness.models import HarnessTaskPack
from vtm_rlm.memory_bridge import summarize_memory_context


def build_phase1_task_prompt(task_pack: HarnessTaskPack, workspace_root: Path) -> str:
    """Build the initial task prompt passed to the vendored RLM runtime."""
    sections = [
        "You are operating as the repository execution engine for a coding benchmark task.",
        f"Workspace root: {workspace_root}",
        "Modify files under the workspace root to solve the task.",
        (
            "Use the VTM memory tools `search_memory` and `expand_memory` when they can "
            "help you ground decisions in retrieved repository memory."
        ),
        "",
        "Task",
        task_pack.task_statement,
    ]
    if task_pack.problem_statement:
        sections.extend(["", "Problem Statement", task_pack.problem_statement])
    if task_pack.hints_text:
        sections.extend(["", "Hints", task_pack.hints_text])
    if task_pack.fail_to_pass_tests:
        sections.extend(["", "Fail-to-Pass Tests", "\n".join(task_pack.fail_to_pass_tests)])
    elif task_pack.failing_tests:
        sections.extend(["", "Failing Tests", "\n".join(task_pack.failing_tests)])
    if task_pack.pass_to_pass_tests:
        sections.extend(["", "Pass-to-Pass Tests", "\n".join(task_pack.pass_to_pass_tests)])
    if task_pack.expected_changed_paths:
        sections.extend(
            ["", "Expected Changed Paths", "\n".join(task_pack.expected_changed_paths)]
        )
    if task_pack.memory_context:
        sections.extend(
            ["", "Preloaded VTM Memory", summarize_memory_context(task_pack.memory_context)]
        )
    sections.extend(
        [
            "",
            "Completion Rules",
            "- Work directly against files in the workspace root.",
            "- Prefer narrow, task-relevant edits.",
            "- Use VTM memory tools when context is ambiguous or likely to drift.",
            "- When the task is complete, return a concise final answer describing what changed.",
        ]
    )
    return "\n".join(sections)
