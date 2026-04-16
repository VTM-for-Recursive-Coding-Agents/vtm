"""Execution helpers for running the vendored upstream RLM against VTM tasks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from vtm.harness.models import HarnessTaskPack
from vtm.memory_items import VisibilityScope
from vtm.services import TransactionalMemoryKernel
from vtm_rlm._vendored import load_rlm_runtime
from vtm_rlm.memory_bridge import VTMMemoryBridge
from vtm_rlm.prompting import build_phase1_task_prompt


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
    custom_tools: dict[str, object] = {
        "WORKSPACE_ROOT": {
            "tool": str(workspace_root),
            "description": "Absolute path to the writable benchmark workspace.",
        },
        "TASK": {
            "tool": task_pack.model_dump(mode="json"),
            "description": "Full coding-task payload for the current benchmark case.",
        },
        "PRELOADED_MEMORY": {
            "tool": [item.model_dump(mode="json") for item in task_pack.memory_context],
            "description": "VTM memory retrieved before execution started.",
        },
        **bridge.custom_tools(),
    }
    backend_kwargs: dict[str, Any] = {"model_name": model_id}
    if base_url:
        backend_kwargs["base_url"] = base_url
        if _is_ollama_base_url(base_url):
            backend_kwargs["completion_extra_body"] = {"think": False}
    if api_key:
        backend_kwargs["api_key"] = api_key

    runtime = rlm_class(
        backend="openai",
        backend_kwargs=backend_kwargs,
        environment="local",
        max_depth=max_depth,
        max_iterations=max_iterations,
        max_timeout=max_timeout_seconds,
        logger=logger,
        custom_tools=custom_tools,
        verbose=False,
    )
    completion = runtime.completion(
        build_phase1_task_prompt(task_pack, workspace_root),
        root_prompt=task_pack.task_statement,
    )

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
            json.dumps(completion.metadata, indent=2, sort_keys=True, default=str),
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
        metadata=completion.metadata or {},
    )


def _is_ollama_base_url(base_url: str) -> bool:
    parsed = urlparse(base_url)
    hostname = (parsed.hostname or "").lower()
    if hostname in {"localhost", "127.0.0.1"} and parsed.port == 11434:
        return True
    return "ollama" in base_url.lower()
