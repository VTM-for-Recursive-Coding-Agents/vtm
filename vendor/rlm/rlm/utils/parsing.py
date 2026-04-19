"""
Parsing utilities for RLM trjaectories.
"""

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rlm.core.types import REPLResult, RLMIteration

if TYPE_CHECKING:
    from rlm.environments.base_env import BaseEnv


_REPL_FENCE_PATTERN = re.compile(r"```repl\s*\n(.*?)\n```", re.DOTALL)
_JSON_FENCE_PATTERN = re.compile(r"```json\s*\n(.*?)\n```", re.DOTALL)
_REPL_FENCE_ONLY_PATTERN = re.compile(r"\A```repl\s*\n(.*?)\n```\s*\Z", re.DOTALL)
_JSON_FENCE_ONLY_PATTERN = re.compile(r"\A```json\s*\n(.*?)\n```\s*\Z", re.DOTALL)


@dataclass(frozen=True)
class ReplExtraction:
    code_blocks: list[str]
    fenced_repl_block_count: int
    json_repl_block_count: int

    @property
    def had_json_repl(self) -> bool:
        return self.json_repl_block_count > 0


def find_code_blocks(text: str) -> list[str]:
    """
    Find REPL code blocks in text wrapped in triple backticks and return List of content(s).
    Returns None if no code blocks are found.
    """
    return extract_repl_blocks(text).code_blocks


def extract_repl_blocks(text: str) -> ReplExtraction:
    """Extract executable REPL code from fenced repl blocks and supported JSON tool calls."""
    results: list[str] = []

    for match in _REPL_FENCE_PATTERN.finditer(text):
        results.append(match.group(1).strip())

    json_repl_count = 0
    for candidate in _iter_json_candidates(text):
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        extracted = _extract_json_repl_code(payload)
        if extracted is None:
            continue
        json_repl_count += 1
        results.append(extracted)

    return ReplExtraction(
        code_blocks=results,
        fenced_repl_block_count=len(results) - json_repl_count,
        json_repl_block_count=json_repl_count,
    )


def _iter_json_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        candidates.append(stripped)
    candidates.extend(match.group(1).strip() for match in _JSON_FENCE_PATTERN.finditer(text))
    return candidates


def _extract_json_repl_code(payload: object) -> str | None:
    if not isinstance(payload, dict):
        return None

    repl_payload = payload.get("repl")
    if isinstance(repl_payload, str):
        return repl_payload
    if isinstance(repl_payload, dict):
        code = repl_payload.get("code")
        if isinstance(code, str):
            return code

    command = payload.get("command")
    tool = payload.get("tool")
    if command != "repl" and tool != "repl":
        return None

    arguments = payload.get("args")
    if not isinstance(arguments, dict):
        arguments = payload.get("arguments")
    if not isinstance(arguments, dict):
        return None
    code = arguments.get("code")
    return code if isinstance(code, str) else None


def is_pure_repl_tool_call(text: str) -> bool:
    """Return True when the full response is only a supported REPL tool call."""
    stripped = text.strip()
    if not stripped:
        return False

    repl_match = _REPL_FENCE_ONLY_PATTERN.fullmatch(stripped)
    if repl_match is not None:
        return bool(repl_match.group(1).strip())

    json_payload: object | None = None
    json_match = _JSON_FENCE_ONLY_PATTERN.fullmatch(stripped)
    if json_match is not None:
        try:
            json_payload = json.loads(json_match.group(1).strip())
        except json.JSONDecodeError:
            return False
    elif stripped.startswith("{") and stripped.endswith("}"):
        try:
            json_payload = json.loads(stripped)
        except json.JSONDecodeError:
            return False

    return _extract_json_repl_code(json_payload) is not None


def find_final_answer(text: str, environment: "BaseEnv | None" = None) -> str | None:
    """
    Find FINAL(...) or FINAL_VAR(...) statement in response and return the final answer string.

    If FINAL_VAR is found and an environment is provided, executes code to retrieve the variable value.
    Returns None if neither pattern is found.

    Args:
        text: The response text to parse
        environment: Optional environment to execute code for FINAL_VAR retrieval

    Returns:
        The final answer string, or None if no final answer pattern is found
    """
    # Check for FINAL_VAR pattern first - must be at start of line
    final_var_pattern = r"^\s*FINAL_VAR\((.*?)\)"
    match = re.search(final_var_pattern, text, re.MULTILINE | re.DOTALL)
    if match:
        variable_name = match.group(1).strip().strip('"').strip("'")
        if environment is not None:
            result = environment.execute_code(f"print(FINAL_VAR({variable_name!r}))")
            final_answer = result.stdout.strip()
            if final_answer == "":
                return None
            # Don't treat FINAL_VAR "variable not found" as final answer (so RLM continues)
            if (
                "Variable '" in final_answer
                and "' not found" in final_answer
                and "FINAL_VAR" in final_answer
            ):
                return None
            return final_answer
        return None

    # Check for FINAL pattern - must be at start of line
    # Use greedy matching to capture content with nested parentheses
    final_pattern = r"^\s*FINAL\((.*)\)\s*$"
    match = re.search(final_pattern, text, re.MULTILINE | re.DOTALL)
    if match:
        return match.group(1).strip()

    return None


def format_iteration(
    iteration: RLMIteration, max_character_length: int = 20000
) -> list[dict[str, str]]:
    """
    Format an RLM iteration (including all code blocks) to append to the message history for
    the prompt of the LM in the next iteration. We also truncate code execution results
    that exceed the max_character_length.

    Args:
        iteration: The iteration to format
        max_character_length: The maximum character length of the result

    Returns:
        A list of messages to add to the next prompt
    """
    messages = [{"role": "assistant", "content": iteration.response}]

    for code_block in iteration.code_blocks:
        code = code_block.code
        result = code_block.result
        result = format_execution_result(result)
        if len(result) > max_character_length:
            result = (
                result[:max_character_length]
                + f"... + [{len(result) - max_character_length} chars...]"
            )

        execution_message = {
            "role": "user",
            "content": f"Code executed:\n```python\n{code}\n```\n\nREPL output:\n{result}",
        }
        messages.append(execution_message)
    return messages


################
# TODO: Remove and refactor these soon
################


def format_execution_result(result: REPLResult) -> str:
    """
    Format the execution result as a string for display.

    Args:
        result: The REPLResult object to format.
    """
    result_parts = []

    if result.stdout:
        result_parts.append(f"\n{result.stdout}")

    if result.stderr:
        result_parts.append(f"\n{result.stderr}")

    # Show some key variables (excluding internal ones)
    important_vars = {}
    for key, value in result.locals.items():
        if not key.startswith("_") and key not in [
            "__builtins__",
            "__name__",
            "__doc__",
        ]:
            # Only show simple types or short representations
            if isinstance(value, (str, int, float, bool, list, dict, tuple)):
                important_vars[key] = ""

    if important_vars:
        result_parts.append(f"REPL variables: {list(important_vars.keys())}\n")

    return "\n\n".join(result_parts) if result_parts else "No output"


def check_for_final_answer(response: str, repl_env, logger) -> str | None:
    """Check if response contains a final answer."""
    # Use the new find_final_answer function which handles both FINAL and FINAL_VAR
    return find_final_answer(response, environment=repl_env)


def convert_context_for_repl(context):
    """
    Convert REPL context to either some
    """
    if isinstance(context, dict):
        context_data = context
        context_str = None
    elif isinstance(context, str):
        context_data = None
        context_str = context
    elif isinstance(context, list):
        if len(context) > 0 and isinstance(context[0], dict):
            if "content" in context[0]:
                context_data = [msg.get("content", "") for msg in context]
            else:
                context_data = context
            context_str = None
        else:
            context_data = context
            context_str = None
    else:
        context_data = context
        context_str = None

    return context_data, context_str
