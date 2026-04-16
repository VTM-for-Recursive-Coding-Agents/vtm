"""Executor implementations that run task packs inside prepared workspaces."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal, Protocol, cast

from vtm.harness.models import ExecutorRequest, ExecutorResult, HarnessTaskPack
from vtm.harness.workspace import PreparedWorkspace
from vtm_rlm.context import RLMRuntimeContext
from vtm_rlm.execution import run_vendored_rlm
from vtm_rlm.writeback import write_success_memory


class BenchmarkExecutor(Protocol):
    """Contract implemented by coding-task executors."""

    def execute(
        self,
        *,
        request: ExecutorRequest,
        prepared_workspace: PreparedWorkspace,
        runtime_context: RLMRuntimeContext | None = None,
    ) -> ExecutorResult: ...


class RLMBenchmarkExecutor:
    """Runs a coding task through the vendored upstream RLM runtime."""

    def __init__(
        self,
        *,
        model_id: str,
        base_url: str | None = None,
        api_key: str | None = None,
        max_iterations: int = 12,
        max_timeout_seconds: int = 600,
        max_depth: int = 2,
    ) -> None:
        if not model_id:
            raise ValueError("RLM benchmark executor requires a non-empty model_id")
        self._model_id = model_id
        self._base_url = base_url
        self._api_key = api_key
        self._max_iterations = max_iterations
        self._max_timeout_seconds = max_timeout_seconds
        self._max_depth = max_depth

    def execute(
        self,
        *,
        request: ExecutorRequest,
        prepared_workspace: PreparedWorkspace,
        runtime_context: RLMRuntimeContext | None = None,
    ) -> ExecutorResult:
        """Execute the task through the vendored RLM runtime and normalize outputs."""
        if runtime_context is None:
            raise ValueError("RLM benchmark executor requires runtime_context")
        artifact_root = prepared_workspace.artifact_root
        rlm_artifact_root = artifact_root / "rlm"
        rlm_artifact_root.mkdir(parents=True, exist_ok=True)
        verification_stdout_path = artifact_root / "final-verification.stdout"
        verification_stderr_path = artifact_root / "final-verification.stderr"
        produced_patch_path = artifact_root / "produced.patch"
        final_git_status_path = artifact_root / "final-git-status.txt"
        task_pack = HarnessTaskPack.model_validate_json(
            Path(request.task_file).read_text(encoding="utf-8")
        )
        scopes = tuple(
            scope
            for scope in (runtime_context.task_scope, runtime_context.durable_scope)
            if scope is not None
        )
        try:
            rlm_result = run_vendored_rlm(
                task_pack=task_pack,
                workspace_root=prepared_workspace.workspace_root,
                artifact_root=rlm_artifact_root,
                model_id=self._model_id,
                kernel=runtime_context.kernel,
                scopes=scopes,
                max_iterations=self._max_iterations,
                max_depth=self._max_depth,
                max_timeout_seconds=self._max_timeout_seconds,
                base_url=self._base_url,
                api_key=self._api_key,
            )

            test_result = None
            if request.test_command:
                test_result = prepared_workspace.driver.run_verification(
                    request.test_command,
                    label="final_verification",
                )
                verification_stdout_path.write_text(test_result.stdout, encoding="utf-8")
                verification_stderr_path.write_text(test_result.stderr, encoding="utf-8")

            produced_patch = prepared_workspace.driver.capture_patch()
            produced_patch_path.write_text(produced_patch, encoding="utf-8")
            produced_patch_digest = hashlib.sha256(produced_patch.encode("utf-8")).hexdigest()
            changed_paths = prepared_workspace.driver.capture_changed_paths()
            final_git_status = prepared_workspace.driver.git_status()
            final_git_status_path.write_text(final_git_status, encoding="utf-8")
            memory_id = None
            if (
                test_result is not None
                and test_result.exit_code == 0
                and not test_result.timed_out
            ):
                memory_id = write_success_memory(
                    kernel=runtime_context.kernel,
                    dependency_builder=runtime_context.dependency_builder,
                    workspace_root=prepared_workspace.workspace_root,
                    task_statement=task_pack.task_statement,
                    case_id=task_pack.case_id,
                    scope=runtime_context.durable_scope,
                    produced_patch_text=produced_patch,
                    run_result=rlm_result,
                )
            usage_summary = dict(rlm_result.usage_summary)
            return ExecutorResult(
                command=("rlm",),
                command_exit_code=0,
                command_stdout_path=rlm_result.response_path,
                command_stderr_path=None,
                attempt_index=request.attempt_index,
                runtime_ms=rlm_result.runtime_ms,
                workspace=request.workspace,
                task_file=request.task_file,
                test_command=request.test_command,
                test_exit_code=test_result.exit_code if test_result is not None else None,
                test_stdout_path=str(verification_stdout_path) if test_result is not None else None,
                test_stderr_path=str(verification_stderr_path) if test_result is not None else None,
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
                    prepared_workspace, "docker_container_id"
                ),
                docker_container_name=_workspace_metadata(
                    prepared_workspace, "docker_container_name"
                ),
                docker_network=_docker_network(prepared_workspace),
                produced_patch_path=str(produced_patch_path),
                produced_patch_digest=produced_patch_digest,
                produced_patch_text=produced_patch,
                produced_changed_paths=changed_paths,
                agent_metrics={
                    "rlm_total_input_tokens": usage_summary.get("total_input_tokens", 0),
                    "rlm_total_output_tokens": usage_summary.get("total_output_tokens", 0),
                    "rlm_total_cost": usage_summary.get("total_cost"),
                },
                agent_artifacts={
                    "rlm_response_path": rlm_result.response_path,
                    "rlm_completion_json_path": rlm_result.completion_json_path,
                    "rlm_trajectory_dir": rlm_result.trajectory_dir or "",
                    **(
                        {"rlm_trajectory_json_path": rlm_result.metadata_json_path}
                        if rlm_result.metadata_json_path is not None
                        else {}
                    ),
                },
                agent_metadata={
                    "agent_status": "completed",
                    "agent_final_message": rlm_result.response,
                    "rlm_model_id": self._model_id,
                    "agent_mode": "vendored_rlm",
                    "rlm_memory_id": memory_id,
                    "rlm_usage_summary": usage_summary,
                    "rlm_metadata": rlm_result.metadata,
                },
            )
        finally:
            prepared_workspace.driver.close()


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
    "RLMBenchmarkExecutor",
]
