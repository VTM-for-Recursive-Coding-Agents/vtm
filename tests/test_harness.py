from __future__ import annotations

from pathlib import Path

from vtm.agents.workspace import LocalWorkspaceDriver as AgentWorkspaceDriver
from vtm.benchmarks.executor import (
    ExecutorResult as BenchmarkExecutorResult,
)
from vtm.benchmarks.executor import (
    SubprocessBenchmarkExecutor as BenchmarkSubprocessExecutor,
)
from vtm.harness.executors import SubprocessBenchmarkExecutor
from vtm.harness.models import (
    ExecutorRequest,
    ExecutorResult,
    HarnessTaskPack,
    TaskMemoryContextItem,
    TraceManifest,
)
from vtm.harness.workspace import LocalWorkspaceDriver


def test_harness_models_round_trip() -> None:
    task_pack = HarnessTaskPack(
        case_id="task-1",
        repo_name="repo",
        commit_pair_id="pair",
        evaluation_backend="local_subprocess",
        base_ref="base",
        head_ref="head",
        task_statement="Fix the bug.",
        expected_changed_paths=("bug.py",),
        touched_paths=("bug.py",),
        retrieval_query="fix bug path",
        test_command=("python", "-m", "pytest"),
        target_patch_digest="deadbeef",
        memory_mode="lexical",
        top_k=5,
        memory_context=(
            TaskMemoryContextItem(
                memory_id="mem-1",
                title="Relevant memory",
                summary="This memory matters.",
                score=0.9,
                status="verified",
                relative_path="bug.py",
                symbol="fix_bug",
                raw_anchor_path="bug.py",
            ),
        ),
        coding_executor="external_command",
    )
    restored_task_pack = HarnessTaskPack.from_json(task_pack.to_json())
    assert restored_task_pack == task_pack

    executor_request = ExecutorRequest(
        case_id="task-1",
        task_file=".benchmarks/task.json",
        workspace=".benchmarks/workspace",
        artifact_root=".benchmarks/artifacts",
        coding_executor="native_agent",
        attempt_index=2,
        command=("python", "worker.py"),
        test_command=("python", "-m", "pytest"),
    )
    restored_request = ExecutorRequest.from_json(executor_request.to_json())
    assert restored_request == executor_request

    executor_result = ExecutorResult(
        command=("native_agent",),
        command_exit_code=0,
        command_stdout_path=None,
        command_stderr_path=None,
        attempt_index=2,
        runtime_ms=1234.5,
        workspace=".benchmarks/workspace",
        task_file=".benchmarks/task.json",
        test_command=("python", "-m", "pytest"),
        produced_patch_text="diff --git a/a b/a\n",
        trace_manifest=TraceManifest(
            session="session.json",
            turns_jsonl="turns.jsonl",
            tool_calls_jsonl="tool_calls.jsonl",
            compactions_jsonl="compactions.jsonl",
            tool_results_dir="tool-results",
        ),
    )
    restored_result = ExecutorResult.from_json(executor_result.to_json())
    assert restored_result == executor_result


def test_harness_shims_reexport_new_modules() -> None:
    assert AgentWorkspaceDriver is LocalWorkspaceDriver
    assert BenchmarkSubprocessExecutor is SubprocessBenchmarkExecutor
    assert BenchmarkExecutorResult is ExecutorResult


def test_local_workspace_driver_preserves_blank_lines(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    driver = LocalWorkspaceDriver(
        workspace_root=workspace,
        artifact_root=tmp_path / "artifacts",
        default_max_output_chars=200,
    )
    try:
        result = driver.run_terminal(
            "python -c \"print('alpha'); print(); print('beta')\"",
        )
    finally:
        driver.close()

    assert result.timed_out is False
    assert "alpha\n\nbeta" in result.output
