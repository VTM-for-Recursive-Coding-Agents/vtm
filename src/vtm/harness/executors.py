"""Executor implementations that run task packs inside prepared workspaces."""

from __future__ import annotations

import hashlib
import shlex
from datetime import datetime
from typing import Protocol

from vtm.adapters.agent_model import AgentModelAdapter
from vtm.agents import (
    AgentRunRequest,
    AgentRunResult,
    AgentRunStatus,
    AgentRuntimeContext,
    TerminalCodingAgent,
)
from vtm.harness.models import ExecutorRequest, ExecutorResult, TraceManifest
from vtm.harness.workspace import PreparedWorkspace


class BenchmarkExecutor(Protocol):
    """Contract implemented by coding-task executors."""

    def execute(
        self,
        *,
        request: ExecutorRequest,
        prepared_workspace: PreparedWorkspace,
        run_request: AgentRunRequest | None = None,
        runtime_context: AgentRuntimeContext | None = None,
    ) -> ExecutorResult: ...


class SubprocessBenchmarkExecutor:
    """Runs the configured command directly inside the prepared workspace."""

    def execute(
        self,
        *,
        request: ExecutorRequest,
        prepared_workspace: PreparedWorkspace,
        run_request: AgentRunRequest | None = None,
        runtime_context: AgentRuntimeContext | None = None,
    ) -> ExecutorResult:
        """Execute a non-agent coding task and collect output artifacts."""
        del run_request, runtime_context
        if not request.command:
            raise ValueError("subprocess benchmark executor requires a command")
        artifact_root = prepared_workspace.artifact_root
        command_stdout_path = artifact_root / "command.stdout"
        command_stderr_path = artifact_root / "command.stderr"
        verification_stdout_path = artifact_root / "final-verification.stdout"
        verification_stderr_path = artifact_root / "final-verification.stderr"
        produced_patch_path = artifact_root / "produced.patch"
        final_git_status_path = artifact_root / "final-git-status.txt"
        try:
            command_result = prepared_workspace.driver.run_terminal(shlex.join(request.command))
            command_stdout_path.write_text(command_result.output, encoding="utf-8")
            command_stderr_path.write_text("", encoding="utf-8")

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
            return ExecutorResult(
                command=request.command,
                command_exit_code=command_result.exit_code,
                command_stdout_path=str(command_stdout_path),
                command_stderr_path=str(command_stderr_path),
                attempt_index=request.attempt_index,
                command_timed_out=command_result.timed_out,
                runtime_ms=command_result.duration_ms,
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
                workspace_backend=prepared_workspace.backend_name,
                produced_patch_path=str(produced_patch_path),
                produced_patch_digest=produced_patch_digest,
                produced_patch_text=produced_patch,
                produced_changed_paths=changed_paths,
            )
        finally:
            prepared_workspace.driver.close()


class NativeAgentBenchmarkExecutor:
    """Runs a coding task through the native single-agent runtime."""

    def __init__(self, *, model_adapter: AgentModelAdapter) -> None:
        """Bind the executor to the model adapter used by the agent."""
        self._model_adapter = model_adapter

    def execute(
        self,
        *,
        request: ExecutorRequest,
        prepared_workspace: PreparedWorkspace,
        run_request: AgentRunRequest | None = None,
        runtime_context: AgentRuntimeContext | None = None,
    ) -> ExecutorResult:
        """Execute the task through `TerminalCodingAgent` and normalize outputs."""
        if run_request is None or runtime_context is None:
            raise ValueError(
                "native agent benchmark executor requires run_request and runtime_context"
            )
        artifact_root = prepared_workspace.artifact_root
        agent_artifact_root = artifact_root / "agent"
        agent_artifact_root.mkdir(parents=True, exist_ok=True)
        verification_stdout_path = artifact_root / "final-verification.stdout"
        verification_stderr_path = artifact_root / "final-verification.stderr"
        produced_patch_path = artifact_root / "produced.patch"
        final_git_status_path = artifact_root / "final-git-status.txt"
        try:
            agent = TerminalCodingAgent(model_adapter=self._model_adapter)
            agent_result = agent.run(
                run_request,
                AgentRuntimeContext(
                    task_file=runtime_context.task_file,
                    workspace_root=runtime_context.workspace_root,
                    artifact_root=agent_artifact_root,
                    task_payload=runtime_context.task_payload,
                    test_command=runtime_context.test_command,
                    workspace_driver=runtime_context.workspace_driver,
                    kernel=runtime_context.kernel,
                    task_scope=runtime_context.task_scope,
                    durable_scope=runtime_context.durable_scope,
                    dependency_builder=runtime_context.dependency_builder,
                ),
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
            trace_manifest = TraceManifest(
                session=agent_result.artifacts["session"],
                turns_jsonl=agent_result.artifacts["turns_jsonl"],
                tool_calls_jsonl=agent_result.artifacts["tool_calls_jsonl"],
                compactions_jsonl=agent_result.artifacts["compactions_jsonl"],
                tool_results_dir=agent_result.artifacts["tool_results_dir"],
            )
            return ExecutorResult(
                command=("native_agent",),
                command_exit_code=0 if agent_result.status is AgentRunStatus.COMPLETED else 1,
                command_stdout_path=None,
                command_stderr_path=None,
                attempt_index=request.attempt_index,
                runtime_ms=self._runtime_ms(agent_result),
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
                workspace_backend=prepared_workspace.backend_name,
                produced_patch_path=str(produced_patch_path),
                produced_patch_digest=produced_patch_digest,
                produced_patch_text=produced_patch,
                produced_changed_paths=changed_paths,
                trace_manifest=trace_manifest,
                agent_metrics={
                    "turn_count": agent_result.turn_count,
                    "tool_call_count": agent_result.tool_call_count,
                    "tool_failure_count": agent_result.tool_failure_count,
                    "terminal_command_count": agent_result.terminal_command_count,
                    "command_timeout_count": agent_result.command_timeout_count,
                    "compaction_count": agent_result.compaction_count,
                    "test_iterations": agent_result.test_iterations,
                    "first_passing_turn": agent_result.first_passing_turn,
                    "memory_write_count": agent_result.memory_write_count,
                    "memory_promotion_count": agent_result.memory_promotion_count,
                    "guardrail_blocks": agent_result.guardrail_blocks,
                },
                agent_artifacts=dict(agent_result.artifacts),
                agent_metadata={
                    "agent_status": agent_result.status.value,
                    "agent_final_message": agent_result.final_message,
                    "agent_model_id": agent_result.model_id,
                    "agent_mode": agent_result.mode.value,
                    **agent_result.metadata,
                },
            )
        finally:
            prepared_workspace.driver.close()

    def _runtime_ms(self, result: AgentRunResult) -> float:
        """Convert agent ISO timestamps into a runtime measurement."""
        started = datetime.fromisoformat(result.started_at)
        completed = datetime.fromisoformat(result.completed_at)
        return (completed - started).total_seconds() * 1000


__all__ = [
    "BenchmarkExecutor",
    "NativeAgentBenchmarkExecutor",
    "SubprocessBenchmarkExecutor",
]
