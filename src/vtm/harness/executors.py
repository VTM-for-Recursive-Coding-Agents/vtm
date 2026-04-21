"""Executor implementations that run task packs inside prepared workspaces."""

from __future__ import annotations

import hashlib
import json
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from vtm.fingerprints import DependencyFingerprint
from vtm.harness.models import ExecutorRequest, ExecutorResult, HarnessTaskPack
from vtm.harness.workspace import PreparedWorkspace
from vtm.memory_items import MemoryItem, VisibilityScope
from vtm.services.memory_kernel import MemoryKernel
from vtm_dspy.config import DSPyOpenRouterConfig
from vtm_dspy.react_agent import VTMReActCodingAgent
from vtm_dspy.tools import memory_tooling_supported


@dataclass(frozen=True)
class ExecutorMemoryRuntime:
    """Live memory handles optionally threaded into one executor run."""

    kernel: MemoryKernel | None = None
    scopes: tuple[VisibilityScope, ...] = ()
    dependency_provider: Callable[[], DependencyFingerprint | None] | None = None
    memory_lookup: Callable[[str], MemoryItem | None] | None = None

    @property
    def enabled(self) -> bool:
        """Whether dynamic VTM memory tools are fully supported."""
        return memory_tooling_supported(
            kernel=self.kernel,
            scopes=self.scopes,
            dependency_provider=self.dependency_provider,
            memory_lookup=self.memory_lookup,
        )


class BenchmarkExecutor(Protocol):
    """Contract implemented by coding-task executors."""

    def execute(
        self,
        *,
        request: ExecutorRequest,
        prepared_workspace: PreparedWorkspace,
        memory_runtime: ExecutorMemoryRuntime | None = None,
    ) -> ExecutorResult: ...


class DSPyReActBenchmarkExecutor:
    """Runs a coding task through the maintained DSPy ReAct workspace agent."""

    def __init__(
        self,
        *,
        model_id: str,
        base_url: str | None = None,
        api_key: str | None = None,
        max_iterations: int = 12,
        command_timeout_seconds: int = 120,
        max_output_chars: int = 20000,
    ) -> None:
        if not model_id:
            raise ValueError("DSPy benchmark executor requires a non-empty model_id")
        self._model_id = model_id
        self._base_url = base_url
        self._api_key = api_key
        self._max_iterations = max_iterations
        self._command_timeout_seconds = command_timeout_seconds
        self._max_output_chars = max_output_chars
        self._active_request: ExecutorRequest | None = None
        self._active_artifact_root = ""
        self._active_workspace_root = ""

    def execute(
        self,
        *,
        request: ExecutorRequest,
        prepared_workspace: PreparedWorkspace,
        memory_runtime: ExecutorMemoryRuntime | None = None,
    ) -> ExecutorResult:
        """Execute the task with workspace tools and a fixed task-pack prompt."""
        try:
            artifact_root = prepared_workspace.artifact_root
            dspy_artifact_root = artifact_root / "dspy-react"
            dspy_artifact_root.mkdir(parents=True, exist_ok=True)
            response_path = dspy_artifact_root / "response.txt"
            trajectory_path = dspy_artifact_root / "trajectory.json"
            runtime_error_path = dspy_artifact_root / "runtime-error.txt"
            verification_stdout_path = artifact_root / "final-verification.stdout"
            verification_stderr_path = artifact_root / "final-verification.stderr"
            produced_patch_path = artifact_root / "produced.patch"
            final_git_status_path = artifact_root / "final-git-status.txt"
            task_pack = HarnessTaskPack.model_validate_json(
                Path(request.task_file).read_text(encoding="utf-8")
            )
            model_config = DSPyOpenRouterConfig.from_env(
                base_url_value=self._base_url,
                api_key_value=self._api_key,
                execution_model_name=self._model_id,
                dspy_model_name=self._model_id,
            )
            resolved_memory_runtime = memory_runtime or ExecutorMemoryRuntime()
            agent = VTMReActCodingAgent(
                kernel=resolved_memory_runtime.kernel,
                scopes=resolved_memory_runtime.scopes,
                enable_memory_tools=resolved_memory_runtime.enabled,
                workspace_root=prepared_workspace.workspace_root,
                dependency_provider=resolved_memory_runtime.dependency_provider,
                memory_lookup=resolved_memory_runtime.memory_lookup,
                model_config=model_config,
                command_timeout_seconds=self._command_timeout_seconds,
                max_output_chars=self._max_output_chars,
                max_iters=self._max_iterations,
            )
            prompt = build_benchmark_task_prompt(task_pack)
            command_exit_code = 0
            command_stderr_path: str | None = None
            response_text = ""
            trajectory: dict[str, Any] = {}
            self._active_request = request
            self._active_artifact_root = str(dspy_artifact_root)
            self._active_workspace_root = str(prepared_workspace.workspace_root)
            try:
                payload = self._run_agent(agent, prompt)
                response_payload = payload.get("response")
                response_text = _coerce_agent_response(response_payload)
                raw_trajectory = payload.get("trajectory")
                trajectory = raw_trajectory if isinstance(raw_trajectory, dict) else {}
            except Exception as exc:
                command_exit_code = 1
                runtime_error_path.write_text(
                    "".join(traceback.format_exception(exc)),
                    encoding="utf-8",
                )
                command_stderr_path = str(runtime_error_path)
                trajectory = {
                    "execution_mode": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            finally:
                self._active_request = None
                self._active_artifact_root = ""
                self._active_workspace_root = ""

            response_path.write_text(response_text, encoding="utf-8")
            trajectory_path.write_text(
                json.dumps(trajectory, indent=2, sort_keys=True),
                encoding="utf-8",
            )

            test_result = None
            if request.test_command:
                test_result = prepared_workspace.driver.run_verification(
                    request.test_command,
                    label=f"dspy_react_attempt_{request.attempt_index:02d}_verification",
                )
                verification_stdout_path.write_text(test_result.stdout, encoding="utf-8")
                verification_stderr_path.write_text(test_result.stderr, encoding="utf-8")

            produced_patch = prepared_workspace.driver.capture_patch()
            changed_paths = prepared_workspace.driver.capture_changed_paths()
            final_git_status = prepared_workspace.driver.git_status()
            produced_patch_path.write_text(produced_patch, encoding="utf-8")
            final_git_status_path.write_text(final_git_status, encoding="utf-8")

            diagnostics = trajectory.get("diagnostics")
            if not isinstance(diagnostics, dict):
                diagnostics = {}
            tool_failure_count = _tool_failure_count(diagnostics)
            return ExecutorResult(
                command=("dspy_react",),
                command_exit_code=command_exit_code,
                command_stdout_path=str(response_path),
                command_stderr_path=command_stderr_path,
                attempt_index=request.attempt_index,
                command_timed_out=False,
                runtime_ms=float(diagnostics.get("total_lm_duration_ms", 0.0) or 0.0),
                workspace=request.workspace,
                task_file=request.task_file,
                test_command=request.test_command,
                test_exit_code=test_result.exit_code if test_result is not None else None,
                test_stdout_path=(
                    str(verification_stdout_path) if test_result is not None else None
                ),
                test_stderr_path=(
                    str(verification_stderr_path) if test_result is not None else None
                ),
                final_verification_runtime_ms=(
                    test_result.duration_ms if test_result is not None else None
                ),
                final_verification_timed_out=(
                    test_result.timed_out if test_result is not None else False
                ),
                final_git_status_path=str(final_git_status_path),
                command_events_path=str(prepared_workspace.command_events_path),
                workspace_backend=_workspace_backend(prepared_workspace),
                docker_image=_workspace_metadata(prepared_workspace, "docker_image"),
                docker_container_id=_workspace_metadata(
                    prepared_workspace,
                    "docker_container_id",
                ),
                docker_container_name=_workspace_metadata(
                    prepared_workspace,
                    "docker_container_name",
                ),
                docker_network=_docker_network(prepared_workspace),
                produced_patch_path=str(produced_patch_path),
                produced_patch_digest=hashlib.sha256(
                    produced_patch.encode("utf-8")
                ).hexdigest(),
                produced_patch_text=produced_patch,
                produced_changed_paths=changed_paths,
                agent_metrics={
                    "prompt_tokens": diagnostics.get("total_prompt_tokens", 0),
                    "completion_tokens": diagnostics.get("total_completion_tokens", 0),
                    "turn_count": diagnostics.get("lm_call_count", 0),
                    "tool_call_count": diagnostics.get("tool_call_count", 0),
                    "tool_failure_count": tool_failure_count,
                    "truncated_lm_call_count": diagnostics.get(
                        "truncated_lm_call_count",
                        0,
                    ),
                },
                agent_artifacts={
                    "dspy_response_path": str(response_path),
                    "dspy_trajectory_path": str(trajectory_path),
                    **(
                        {"runtime_error_path": str(runtime_error_path)}
                        if command_stderr_path is not None
                        else {}
                    ),
                },
                agent_metadata={
                    "agent_status": "completed" if command_exit_code == 0 else "failed",
                    "agent_final_message": response_text,
                    "agent_mode": "dspy_react",
                    "execution_model_id": self._model_id,
                    "execution_mode": trajectory.get("execution_mode"),
                    "fallback_error": trajectory.get("fallback_error"),
                    "memory_context_count": len(task_pack.memory_context),
                    "memory_tools_enabled": agent.memory_tools.enabled,
                    "tool_names": list(agent.tool_names()),
                },
            )
        finally:
            prepared_workspace.driver.close()

    def _run_agent(self, agent: VTMReActCodingAgent, prompt: str) -> dict[str, Any]:
        return agent.run(prompt)


def build_benchmark_task_prompt(task_pack: HarnessTaskPack) -> str:
    """Render the model-visible task pack used for maintained coding runs."""
    sections = [
        "You are editing the repository workspace to fix the task.",
        (
            "Use workspace tools to inspect files, change code, run the visible tests, "
            "and stop when the task is complete."
        ),
        "Apply changes directly in the workspace. Do not print a patch in the final response.",
        "",
        f"Case ID: {task_pack.case_id}",
        f"Task: {task_pack.task_statement.strip()}",
    ]
    if task_pack.problem_statement:
        sections.extend(["", "Problem Statement:", task_pack.problem_statement.strip()])
    if task_pack.hints_text:
        sections.extend(["", "Hints:", task_pack.hints_text.strip()])
    if task_pack.fail_to_pass_tests:
        sections.extend(
            ["", "Fail-To-Pass Tests:", *[f"- {item}" for item in task_pack.fail_to_pass_tests]]
        )
    if task_pack.pass_to_pass_tests:
        sections.extend(
            ["", "Pass-To-Pass Tests:", *[f"- {item}" for item in task_pack.pass_to_pass_tests]]
        )
    if task_pack.test_command:
        sections.extend(["", "Visible Test Command:", " ".join(task_pack.test_command)])
    if task_pack.localization_notes:
        sections.extend(
            ["", "Localization Notes:", *[f"- {item}" for item in task_pack.localization_notes]]
        )
    if task_pack.verifier_output:
        sections.extend(["", "Visible Failure Output:", task_pack.verifier_output.strip()])
    if task_pack.memory_context:
        sections.append("")
        sections.append("Retrieved Memory Context:")
        for index, item in enumerate(task_pack.memory_context, start=1):
            header = f"{index}. {item.title}"
            if item.relative_path and item.symbol:
                header += f" ({item.relative_path}::{item.symbol})"
            elif item.relative_path:
                header += f" ({item.relative_path})"
            sections.append(header)
            sections.append(f"summary: {item.summary}")
            if item.relevance_reason:
                sections.append(f"reason: {item.relevance_reason}")
    return "\n".join(sections).strip()


def _coerce_agent_response(payload: Any) -> str:
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        for key in ("response", "answer", "output", "text"):
            value = payload.get(key)
            if isinstance(value, str):
                return value
        return json.dumps(payload, indent=2, sort_keys=True)
    return str(payload or "")


def _tool_failure_count(diagnostics: dict[str, Any]) -> int:
    tool_calls = diagnostics.get("tool_calls")
    if not isinstance(tool_calls, list):
        return 0
    return sum(
        1
        for item in tool_calls
        if isinstance(item, dict) and str(item.get("exception") or "").strip()
    )


def _workspace_metadata(
    prepared_workspace: PreparedWorkspace,
    key: str,
) -> str | None:
    return prepared_workspace.metadata.get(key)


def _workspace_backend(
    prepared_workspace: PreparedWorkspace,
) -> Literal["local_workspace", "docker_workspace"]:
    return prepared_workspace.backend_name


def _docker_network(
    prepared_workspace: PreparedWorkspace,
) -> Literal["none", "bridge"] | None:
    value = prepared_workspace.metadata.get("docker_network")
    return cast(Literal["none", "bridge"] | None, value)


__all__ = [
    "BenchmarkExecutor",
    "ExecutorMemoryRuntime",
    "DSPyReActBenchmarkExecutor",
    "build_benchmark_task_prompt",
]
