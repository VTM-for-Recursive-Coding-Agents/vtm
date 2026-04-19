"""Execution helpers for running the vendored upstream RLM against VTM tasks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vtm.harness.models import HarnessTaskPack
from vtm.memory_items import VisibilityScope
from vtm.services import TransactionalMemoryKernel
from vtm_rlm._vendored import ensure_vendored_rlm_on_path, load_rlm_runtime
from vtm_rlm.memory_bridge import VTMMemoryBridge
from vtm_rlm.prompting import (
    CODING_RLM_SYSTEM_PROMPT,
    build_phase1_task_prompt,
    model_visible_task_pack,
)


@dataclass(frozen=True)
class VendoredRLMRunResult:
    """Normalized result returned by one vendored-RLM execution."""

    response: str
    runtime_ms: float
    response_path: str
    completion_json_path: str
    metadata_json_path: str | None
    trajectory_dir: str | None
    usage_summary: dict[str, Any]
    metadata: dict[str, Any]


def run_vendored_rlm(
    *,
    task_pack: HarnessTaskPack,
    workspace_root: Path,
    artifact_root: Path,
    model_id: str,
    kernel: TransactionalMemoryKernel | None,
    scopes: tuple[VisibilityScope, ...],
    max_iterations: int,
    max_depth: int,
    max_timeout_seconds: int,
    base_url: str | None = None,
    api_key: str | None = None,
) -> VendoredRLMRunResult:
    """Execute the vendored upstream RLM with VTM memory tools enabled."""
    rlm_class, rlm_logger_class = load_rlm_runtime()
    artifact_root.mkdir(parents=True, exist_ok=True)
    trajectory_dir = artifact_root / "trajectory"
    trajectory_dir.mkdir(parents=True, exist_ok=True)

    logger = rlm_logger_class(log_dir=str(trajectory_dir))
    bridge = VTMMemoryBridge(kernel=kernel, scopes=scopes)
    model_task_pack = model_visible_task_pack(task_pack)
    custom_tools: dict[str, object] = {
        "WORKSPACE_ROOT": {
            "tool": str(workspace_root),
            "description": "Absolute path to the writable benchmark workspace.",
        },
        "TASK": {
            "tool": model_task_pack.model_dump(mode="json"),
            "description": "Full coding-task payload for the current benchmark case.",
        },
        **bridge.custom_tools(),
    }
    backend_kwargs: dict[str, Any] = {"model_name": model_id}
    if base_url:
        backend_kwargs["base_url"] = base_url
    if api_key:
        backend_kwargs["api_key"] = api_key

    runtime = rlm_class(
        backend="openai",
        backend_kwargs=backend_kwargs,
        environment="local",
        max_depth=max_depth,
        max_iterations=max_iterations,
        max_timeout=max_timeout_seconds,
        custom_system_prompt=CODING_RLM_SYSTEM_PROMPT,
        logger=logger,
        custom_tools=custom_tools,
        verbose=False,
    )
    completion = runtime.completion(
        build_phase1_task_prompt(task_pack, workspace_root),
        root_prompt=None,
    )
    enriched_metadata = _enrich_completion_metadata(
        metadata=completion.metadata,
        response=completion.response,
        usage_summary=completion.usage_summary.to_dict(),
    )
    completion.metadata = enriched_metadata

    response_path = artifact_root / "response.txt"
    response_path.write_text(completion.response, encoding="utf-8")
    completion_json_path = artifact_root / "completion.json"
    completion_json_path.write_text(
        json.dumps(completion.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    metadata_json_path: Path | None = None
    if completion.metadata is not None:
        metadata_json_path = artifact_root / "trajectory.json"
        metadata_json_path.write_text(
            json.dumps(enriched_metadata, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
    return VendoredRLMRunResult(
        response=completion.response,
        runtime_ms=completion.execution_time * 1000.0,
        response_path=str(response_path),
        completion_json_path=str(completion_json_path),
        metadata_json_path=str(metadata_json_path) if metadata_json_path is not None else None,
        trajectory_dir=str(trajectory_dir),
        usage_summary=completion.usage_summary.to_dict(),
        metadata=enriched_metadata or {},
    )


def _enrich_completion_metadata(
    *,
    metadata: dict[str, Any] | None,
    response: str,
    usage_summary: dict[str, Any],
) -> dict[str, Any]:
    enriched = dict(metadata or {})
    diagnostics = _summarize_execution_diagnostics(enriched, response=response)
    diagnostics["usage_missing"] = _usage_missing(usage_summary)
    enriched["vtm_execution_diagnostics"] = diagnostics
    return enriched


def _summarize_execution_diagnostics(
    metadata: dict[str, Any],
    *,
    response: str,
) -> dict[str, Any]:
    final_response_was_tool_call = _final_response_was_tool_call(response)
    iterations = metadata.get("iterations")
    if not isinstance(iterations, list):
        return {
            "rlm_iteration_count": 0,
            "rlm_executed_repl_block_count": 0,
            "rlm_detected_json_repl_count": 0,
            "rlm_final_response_had_json_repl": False,
            "final_response_was_tool_call": final_response_was_tool_call,
            "tool_failure_count": 0,
        }

    iteration_count = 0
    executed_repl_block_count = 0
    detected_json_repl_count = 0
    tool_failure_count = 0
    final_response_had_json_repl = False

    for iteration in iterations:
        if not isinstance(iteration, dict):
            continue
        iteration_count += 1
        code_blocks = iteration.get("code_blocks")
        if isinstance(code_blocks, list):
            executed_repl_block_count += len(code_blocks)
            for code_block in code_blocks:
                if not isinstance(code_block, dict):
                    continue
                result = code_block.get("result")
                if isinstance(result, dict) and str(result.get("stderr", "")).strip():
                    tool_failure_count += 1
        action_metadata = iteration.get("action_metadata")
        json_count = 0
        if isinstance(action_metadata, dict):
            try:
                json_count = int(action_metadata.get("json_repl_block_count", 0) or 0)
            except (TypeError, ValueError):
                json_count = 0
        detected_json_repl_count += json_count
        final_response_had_json_repl = json_count > 0

    return {
        "rlm_iteration_count": iteration_count,
        "rlm_executed_repl_block_count": executed_repl_block_count,
        "rlm_detected_json_repl_count": detected_json_repl_count,
        "rlm_final_response_had_json_repl": final_response_had_json_repl,
        "final_response_was_tool_call": final_response_was_tool_call,
        "tool_failure_count": tool_failure_count,
    }


def _usage_missing(usage_summary: dict[str, Any]) -> bool:
    model_summaries = usage_summary.get("model_usage_summaries")
    if not isinstance(model_summaries, dict):
        return False
    return any(
        isinstance(summary, dict) and bool(summary.get("usage_missing"))
        for summary in model_summaries.values()
    )


def _final_response_was_tool_call(response: str) -> bool:
    ensure_vendored_rlm_on_path()
    from rlm.utils.parsing import is_pure_repl_tool_call

    return is_pure_repl_tool_call(response)
