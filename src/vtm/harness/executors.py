"""Executor implementations that run task packs inside prepared workspaces."""

from __future__ import annotations

import hashlib
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from vtm.harness.models import ExecutorRequest, ExecutorResult, HarnessTaskPack
from vtm.harness.workspace import PreparedWorkspace
from vtm_rlm.context import RLMRuntimeContext
from vtm_rlm.execution import VendoredRLMRunResult, run_vendored_rlm
from vtm_rlm.prompting import model_visible_task_pack
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


@dataclass(frozen=True)
class _PhaseRunSnapshot:
    phase_name: str
    used_memory: bool
    rlm_result: VendoredRLMRunResult | None
    test_result: Any | None
    produced_patch: str
    changed_paths: tuple[str, ...]
    final_git_status: str
    runtime_error: str | None
    corrective_retry_used: bool = False
    corrective_retry_reason: str | None = None

    @property
    def passed(self) -> bool:
        return (
            self.test_result is not None
            and self.test_result.exit_code == 0
            and not self.test_result.timed_out
        )


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
        use_memory_fallback = runtime_context.kernel is not None and bool(task_pack.memory_context)
        try:
            phase_runs: list[_PhaseRunSnapshot] = []
            grounding_task_pack = (
                task_pack.model_copy(update={"memory_context": ()})
                if use_memory_fallback
                else task_pack
            )
            grounding_phase = self._run_phase(
                phase_name="grounding",
                task_pack=grounding_task_pack,
                request=request,
                prepared_workspace=prepared_workspace,
                artifact_root=rlm_artifact_root / "phase-1-grounding"
                if use_memory_fallback
                else rlm_artifact_root,
                kernel=None if use_memory_fallback else runtime_context.kernel,
                scopes=() if use_memory_fallback else scopes,
                max_iterations=(
                    min(self._max_iterations, 2)
                    if use_memory_fallback
                    else self._max_iterations
                ),
                max_timeout_seconds=(
                    min(self._max_timeout_seconds, 120)
                    if use_memory_fallback
                    else self._max_timeout_seconds
                ),
                allow_runtime_failure=use_memory_fallback,
            )
            phase_runs.append(grounding_phase)
            final_phase = grounding_phase
            if use_memory_fallback and not grounding_phase.passed:
                memory_phase = self._run_phase(
                    phase_name="memory_fallback",
                    task_pack=task_pack,
                    request=request,
                    prepared_workspace=prepared_workspace,
                    artifact_root=rlm_artifact_root / "phase-2-memory",
                    kernel=runtime_context.kernel,
                    scopes=scopes,
                    max_iterations=self._max_iterations,
                    max_timeout_seconds=self._max_timeout_seconds,
                    allow_runtime_failure=False,
                )
                phase_runs.append(memory_phase)
                final_phase = memory_phase
            self._write_final_artifacts(
                final_phase=final_phase,
                produced_patch_path=produced_patch_path,
                final_git_status_path=final_git_status_path,
                verification_stdout_path=verification_stdout_path,
                verification_stderr_path=verification_stderr_path,
            )
            memory_id = None
            if (
                final_phase.test_result is not None
                and final_phase.test_result.exit_code == 0
                and not final_phase.test_result.timed_out
                and final_phase.rlm_result is not None
            ):
                memory_id = write_success_memory(
                    kernel=runtime_context.kernel,
                    dependency_builder=runtime_context.dependency_builder,
                    workspace_root=prepared_workspace.workspace_root,
                    task_statement=task_pack.task_statement,
                    case_id=task_pack.case_id,
                    scope=runtime_context.durable_scope,
                    produced_patch_text=final_phase.produced_patch,
                    run_result=final_phase.rlm_result,
                )
            final_result = final_phase.rlm_result
            usage_summary = dict(final_result.usage_summary) if final_result is not None else {}
            phase_metadata = [
                {
                    "phase_name": phase.phase_name,
                    "used_memory": phase.used_memory,
                    "passed": phase.passed,
                    "runtime_error": phase.runtime_error,
                    "corrective_retry_used": phase.corrective_retry_used,
                    "corrective_retry_reason": phase.corrective_retry_reason,
                    "produced_changed_paths": list(phase.changed_paths),
                    "response_path": (
                        phase.rlm_result.response_path if phase.rlm_result is not None else None
                    ),
                    "completion_json_path": (
                        phase.rlm_result.completion_json_path
                        if phase.rlm_result is not None
                        else None
                    ),
                }
                for phase in phase_runs
            ]
            return ExecutorResult(
                command=("rlm",),
                command_exit_code=0,
                command_stdout_path=(
                    final_result.response_path if final_result is not None else None
                ),
                command_stderr_path=None,
                attempt_index=request.attempt_index,
                runtime_ms=sum(
                    phase.rlm_result.runtime_ms
                    for phase in phase_runs
                    if phase.rlm_result is not None
                ),
                workspace=request.workspace,
                task_file=request.task_file,
                test_command=request.test_command,
                test_exit_code=(
                    final_phase.test_result.exit_code
                    if final_phase.test_result is not None
                    else None
                ),
                test_stdout_path=(
                    str(verification_stdout_path)
                    if final_phase.test_result is not None
                    else None
                ),
                test_stderr_path=(
                    str(verification_stderr_path)
                    if final_phase.test_result is not None
                    else None
                ),
                final_verification_runtime_ms=(
                    final_phase.test_result.duration_ms
                    if final_phase.test_result is not None
                    else None
                ),
                final_verification_timed_out=(
                    final_phase.test_result.timed_out
                    if final_phase.test_result is not None
                    else False
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
                produced_patch_digest=hashlib.sha256(
                    final_phase.produced_patch.encode("utf-8")
                ).hexdigest(),
                produced_patch_text=final_phase.produced_patch,
                produced_changed_paths=final_phase.changed_paths,
                agent_metrics={
                    "rlm_total_input_tokens": usage_summary.get("total_input_tokens", 0),
                    "rlm_total_output_tokens": usage_summary.get("total_output_tokens", 0),
                    "rlm_total_cost": usage_summary.get("total_cost"),
                },
                agent_artifacts={
                    "rlm_response_path": (
                        final_result.response_path if final_result is not None else ""
                    ),
                    "rlm_completion_json_path": (
                        final_result.completion_json_path if final_result is not None else ""
                    ),
                    "rlm_trajectory_dir": (
                        final_result.trajectory_dir or "" if final_result is not None else ""
                    ),
                    **(
                        {"rlm_trajectory_json_path": final_result.metadata_json_path}
                        if final_result is not None
                        and final_result.metadata_json_path is not None
                        else {}
                    ),
                },
                agent_metadata={
                    "agent_status": "completed" if final_result is not None else "failed",
                    "agent_final_message": (
                        final_result.response if final_result is not None else ""
                    ),
                    "rlm_model_id": self._model_id,
                    "agent_mode": "vendored_rlm",
                    "rlm_memory_id": memory_id,
                    "rlm_usage_summary": usage_summary,
                    "rlm_metadata": final_result.metadata if final_result is not None else {},
                    "rlm_execution_strategy": (
                        "ground_then_memory" if use_memory_fallback else "single_pass"
                    ),
                    "rlm_phases": phase_metadata,
                },
            )
        finally:
            prepared_workspace.driver.close()

    def _run_phase(
        self,
        *,
        phase_name: str,
        task_pack: HarnessTaskPack,
        request: ExecutorRequest,
        prepared_workspace: PreparedWorkspace,
        artifact_root: Path,
        kernel: Any,
        scopes: tuple[Any, ...],
        max_iterations: int,
        max_timeout_seconds: int,
        allow_runtime_failure: bool,
    ) -> _PhaseRunSnapshot:
        artifact_root.mkdir(parents=True, exist_ok=True)
        first_attempt = self._execute_phase_attempt(
            phase_name=phase_name,
            task_pack=task_pack,
            request=request,
            prepared_workspace=prepared_workspace,
            artifact_root=artifact_root,
            kernel=kernel,
            scopes=scopes,
            max_iterations=max_iterations,
            max_timeout_seconds=max_timeout_seconds,
            allow_runtime_failure=allow_runtime_failure,
        )
        corrective_reason = self._corrective_retry_reason(first_attempt)
        if corrective_reason is None:
            return first_attempt
        retry_task_pack = self._build_corrective_retry_task_pack(
            task_pack=task_pack,
            corrective_reason=corrective_reason,
            phase_snapshot=first_attempt,
        )
        retry_root = artifact_root / "repair-retry"
        retry_attempt = self._execute_phase_attempt(
            phase_name=f"{phase_name}_repair_retry",
            task_pack=retry_task_pack,
            request=request,
            prepared_workspace=prepared_workspace,
            artifact_root=retry_root,
            kernel=kernel,
            scopes=scopes,
            max_iterations=min(max_iterations, 2),
            max_timeout_seconds=max_timeout_seconds,
            allow_runtime_failure=allow_runtime_failure,
        )
        return _PhaseRunSnapshot(
            phase_name=retry_attempt.phase_name,
            used_memory=retry_attempt.used_memory,
            rlm_result=retry_attempt.rlm_result,
            test_result=retry_attempt.test_result,
            produced_patch=retry_attempt.produced_patch,
            changed_paths=retry_attempt.changed_paths,
            final_git_status=retry_attempt.final_git_status,
            runtime_error=retry_attempt.runtime_error,
            corrective_retry_used=True,
            corrective_retry_reason=corrective_reason,
        )

    def _execute_phase_attempt(
        self,
        *,
        phase_name: str,
        task_pack: HarnessTaskPack,
        request: ExecutorRequest,
        prepared_workspace: PreparedWorkspace,
        artifact_root: Path,
        kernel: Any,
        scopes: tuple[Any, ...],
        max_iterations: int,
        max_timeout_seconds: int,
        allow_runtime_failure: bool,
    ) -> _PhaseRunSnapshot:
        runtime_error: str | None = None
        rlm_result: VendoredRLMRunResult | None = None
        try:
            rlm_result = run_vendored_rlm(
                task_pack=task_pack,
                workspace_root=prepared_workspace.workspace_root,
                artifact_root=artifact_root,
                model_id=self._model_id,
                kernel=kernel,
                scopes=scopes,
                max_iterations=max_iterations,
                max_depth=self._max_depth,
                max_timeout_seconds=max_timeout_seconds,
                base_url=self._base_url,
                api_key=self._api_key,
            )
        except Exception as exc:
            if not allow_runtime_failure:
                raise
            runtime_error = f"{type(exc).__name__}: {exc}"
            (artifact_root / "runtime-error.txt").write_text(
                "".join(traceback.format_exception(exc)),
                encoding="utf-8",
            )

        test_result = None
        if request.test_command:
            test_result = prepared_workspace.driver.run_verification(
                request.test_command,
                label=f"{phase_name}_verification",
            )
            (artifact_root / "verification.stdout").write_text(
                test_result.stdout,
                encoding="utf-8",
            )
            (artifact_root / "verification.stderr").write_text(
                test_result.stderr,
                encoding="utf-8",
            )

        produced_patch = prepared_workspace.driver.capture_patch()
        changed_paths = prepared_workspace.driver.capture_changed_paths()
        final_git_status = prepared_workspace.driver.git_status()
        (artifact_root / "produced.patch").write_text(produced_patch, encoding="utf-8")
        (artifact_root / "git-status.txt").write_text(final_git_status, encoding="utf-8")
        return _PhaseRunSnapshot(
            phase_name=phase_name,
            used_memory=kernel is not None,
            rlm_result=rlm_result,
            test_result=test_result,
            produced_patch=produced_patch,
            changed_paths=changed_paths,
            final_git_status=final_git_status,
            runtime_error=runtime_error,
        )

    def _corrective_retry_reason(
        self,
        phase_snapshot: _PhaseRunSnapshot,
    ) -> str | None:
        if phase_snapshot.passed or phase_snapshot.rlm_result is None:
            return None
        if not phase_snapshot.changed_paths:
            return "Previous attempt did not modify any repository files."
        if not phase_snapshot.produced_patch.strip():
            return "Previous attempt did not produce a usable patch."
        if phase_snapshot.test_result is not None and phase_snapshot.test_result.exit_code != 0:
            return "Previous attempt changed files but verification still failed."
        return None

    def _build_corrective_retry_task_pack(
        self,
        *,
        task_pack: HarnessTaskPack,
        corrective_reason: str,
        phase_snapshot: _PhaseRunSnapshot,
    ) -> HarnessTaskPack:
        visible_task_pack = model_visible_task_pack(task_pack)
        verification_excerpt = ""
        if phase_snapshot.test_result is not None and phase_snapshot.test_result.stderr:
            verification_excerpt = phase_snapshot.test_result.stderr.strip().splitlines()[0][:200]
        guidance_lines = [
            "Corrective Retry",
            corrective_reason,
            (
                "You must modify the repository files directly and leave a valid patch in "
                "the workspace."
            ),
            "Do not stop at explaining the fix.",
        ]
        if visible_task_pack.expected_changed_paths:
            guidance_lines.append(
                "Focus on these files: " + ", ".join(visible_task_pack.expected_changed_paths)
            )
        if verification_excerpt:
            guidance_lines.append("Latest verifier signal: " + verification_excerpt)
        guidance = "\n".join(guidance_lines)
        updated_hint = visible_task_pack.hints_text or ""
        if updated_hint:
            updated_hint = f"{updated_hint}\n\n{guidance}"
        else:
            updated_hint = guidance
        return task_pack.model_copy(
            update={
                "hints_text": updated_hint,
                "task_statement": f"{visible_task_pack.task_statement}\n\n{guidance}",
            }
        )

    def _write_final_artifacts(
        self,
        *,
        final_phase: _PhaseRunSnapshot,
        produced_patch_path: Path,
        final_git_status_path: Path,
        verification_stdout_path: Path,
        verification_stderr_path: Path,
    ) -> None:
        produced_patch_path.write_text(final_phase.produced_patch, encoding="utf-8")
        final_git_status_path.write_text(final_phase.final_git_status, encoding="utf-8")
        if final_phase.test_result is not None:
            verification_stdout_path.write_text(final_phase.test_result.stdout, encoding="utf-8")
            verification_stderr_path.write_text(final_phase.test_result.stderr, encoding="utf-8")
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
