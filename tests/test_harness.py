from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from vtm.harness.executors import DSPyReActBenchmarkExecutor, ExecutorMemoryRuntime
from vtm.harness.models import (
    ExecutorRequest,
    ExecutorResult,
    HarnessTaskPack,
    TaskMemoryContextItem,
)
from vtm.harness.workspace import LocalWorkspaceBackend, LocalWorkspaceDriver, PreparedWorkspace
from vtm.harness.workspace_docker import DockerWorkspaceBackend, DockerWorkspaceDriver


def _run(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        list(args),
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _build_repo(repo: Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    _run(repo, "git", "init", "-b", "main")
    _run(repo, "git", "config", "user.name", "VTM Harness Tests")
    _run(repo, "git", "config", "user.email", "vtm-harness@example.com")
    (repo / "hello.txt").write_text("hello\n", encoding="utf-8")
    _run(repo, "git", "add", "hello.txt")
    _run(repo, "git", "commit", "-m", "base")


def _build_repo_with_second_commit(repo: Path) -> tuple[str, str]:
    _build_repo(repo)
    base = _run(repo, "git", "rev-parse", "HEAD")
    (repo / "hello.txt").write_text("hello from head\n", encoding="utf-8")
    _run(repo, "git", "add", "hello.txt")
    _run(repo, "git", "commit", "-m", "head")
    head = _run(repo, "git", "rev-parse", "HEAD")
    return base, head


def _build_executor_fixture(tmp_path: Path) -> tuple[ExecutorRequest, PreparedWorkspace]:
    workspace_root = tmp_path / "workspace"
    artifact_root = tmp_path / "artifacts"
    _build_repo(workspace_root)
    artifact_root.mkdir(parents=True, exist_ok=True)
    driver = LocalWorkspaceDriver(
        workspace_root=workspace_root,
        artifact_root=artifact_root,
        default_max_output_chars=200,
    )
    prepared_workspace = PreparedWorkspace(
        workspace_root=workspace_root,
        artifact_root=artifact_root,
        backend_name="local_workspace",
        attempt_index=1,
        command_events_path=driver.command_events_path,
        driver=driver,
    )
    task_pack = HarnessTaskPack(
        case_id="executor-memory-tools",
        repo_name="repo",
        commit_pair_id="pair",
        evaluation_backend="local_subprocess",
        base_ref="HEAD",
        head_ref="HEAD",
        task_statement="Inspect the repository and finish.",
        execution_style="shell_command",
        target_patch_digest="deadbeef",
        memory_mode="verified_lexical",
        top_k=3,
    )
    task_file = artifact_root / "task-pack.json"
    task_file.write_text(task_pack.to_json(), encoding="utf-8")
    request = ExecutorRequest(
        case_id=task_pack.case_id,
        task_file=str(task_file),
        workspace=str(workspace_root),
        artifact_root=str(artifact_root),
        attempt_index=1,
    )
    return request, prepared_workspace


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
        execution_style="shell_command",
        test_command=("python", "-m", "pytest"),
        target_patch_digest="deadbeef",
        memory_mode="verified_lexical",
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
                matched_terms=("fix_bug", "bug"),
                matched_fields=("title", "payload"),
                relevance_reason="matched lexical overlap",
            ),
        ),
    )
    restored_task_pack = HarnessTaskPack.from_json(task_pack.to_json())
    assert restored_task_pack == task_pack

    executor_request = ExecutorRequest(
        case_id="task-1",
        task_file=".benchmarks/task.json",
        workspace=".benchmarks/workspace",
        artifact_root=".benchmarks/artifacts",
        attempt_index=2,
        test_command=("python", "-m", "pytest"),
    )
    restored_request = ExecutorRequest.from_json(executor_request.to_json())
    assert restored_request == executor_request

    executor_result = ExecutorResult(
        command=("dspy_react",),
        command_exit_code=0,
        command_stdout_path=None,
        command_stderr_path=None,
        attempt_index=2,
        runtime_ms=1234.5,
        workspace=".benchmarks/workspace",
        task_file=".benchmarks/task.json",
        test_command=("python", "-m", "pytest"),
        workspace_backend="docker_workspace",
        docker_image="python:3.12",
        docker_container_id="fake-container",
        docker_container_name="fake-name",
        docker_network="none",
        produced_patch_text="diff --git a/a b/a\n",
    )
    restored_result = ExecutorResult.from_json(executor_result.to_json())
    assert restored_result == executor_result


def test_dspy_executor_exposes_dynamic_memory_tools_when_runtime_is_configured(
    tmp_path: Path,
    kernel,
    metadata_store,
    scope,
    dep_fp,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, prepared_workspace = _build_executor_fixture(tmp_path)
    captured: dict[str, object] = {}

    def fake_run(self, agent, prompt: str):  # noqa: ANN001
        del self, prompt
        captured["tool_names"] = agent.tool_names()
        captured["memory_tools_enabled"] = agent.memory_tools.enabled
        return {
            "response": {"response": "done"},
            "trajectory": {
                "execution_mode": "react",
                "diagnostics": {
                    "lm_call_count": 1,
                    "tool_call_count": 0,
                    "total_lm_duration_ms": 1.0,
                    "total_prompt_tokens": 1,
                    "total_completion_tokens": 1,
                    "truncated_lm_call_count": 0,
                },
            },
        }

    monkeypatch.setattr(DSPyReActBenchmarkExecutor, "_run_agent", fake_run)

    result = DSPyReActBenchmarkExecutor(model_id="fake-model").execute(
        request=request,
        prepared_workspace=prepared_workspace,
        memory_runtime=ExecutorMemoryRuntime(
            kernel=kernel,
            scopes=(scope,),
            dependency_provider=lambda: dep_fp,
            memory_lookup=metadata_store.get_memory_item,
        ),
    )

    assert captured["memory_tools_enabled"] is True
    assert "search_verified_memory" in captured["tool_names"]
    assert "search_naive_memory" in captured["tool_names"]
    assert "expand_memory_evidence" in captured["tool_names"]
    assert "verify_memory" in captured["tool_names"]
    assert result.agent_metadata["memory_tools_enabled"] is True


def test_dspy_executor_falls_back_cleanly_when_memory_runtime_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, prepared_workspace = _build_executor_fixture(tmp_path)
    captured: dict[str, object] = {}

    def fake_run(self, agent, prompt: str):  # noqa: ANN001
        del self, prompt
        captured["tool_names"] = agent.tool_names()
        captured["memory_tools_enabled"] = agent.memory_tools.enabled
        return {
            "response": {"response": "done"},
            "trajectory": {
                "execution_mode": "react",
                "diagnostics": {
                    "lm_call_count": 1,
                    "tool_call_count": 0,
                    "total_lm_duration_ms": 1.0,
                    "total_prompt_tokens": 1,
                    "total_completion_tokens": 1,
                    "truncated_lm_call_count": 0,
                },
            },
        }

    monkeypatch.setattr(DSPyReActBenchmarkExecutor, "_run_agent", fake_run)

    result = DSPyReActBenchmarkExecutor(model_id="fake-model").execute(
        request=request,
        prepared_workspace=prepared_workspace,
        memory_runtime=ExecutorMemoryRuntime(),
    )

    assert captured["memory_tools_enabled"] is False
    assert "search_verified_memory" not in captured["tool_names"]
    assert "verify_memory" not in captured["tool_names"]
    assert result.agent_metadata["memory_tools_enabled"] is False


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


def test_docker_workspace_backend_prepares_isolated_workspace(
    tmp_path: Path,
    fake_docker_binary: Path,
) -> None:
    repo_root = tmp_path / "repo"
    _build_repo(repo_root)
    backend = DockerWorkspaceBackend(
        docker_binary=str(fake_docker_binary),
        docker_image="python:3.12",
    )
    prepared = backend.prepare_workspace(
        case_id="shell-case",
        attempt_index=1,
        repo_root=repo_root,
        base_ref="HEAD",
        output_root=tmp_path / "outputs",
        mode="no_memory",
        command_timeout_seconds=30,
        max_output_chars=200,
    )

    try:
        assert isinstance(prepared.driver, DockerWorkspaceDriver)
        assert prepared.backend_name == "docker_workspace"
        assert prepared.metadata["docker_image"] == "python:3.12"
        assert prepared.metadata["docker_network"] == "none"
        assert prepared.metadata["docker_read_only_rootfs"] == "true"
        assert prepared.metadata["docker_pids_limit"] == "256"
        assert prepared.metadata["docker_memory_limit"] == "2g"
        assert prepared.metadata["docker_cpu_limit"] == "2"
        state_path = (
            tmp_path / "fake-docker-state" / (f"{prepared.metadata['docker_container_name']}.json")
        )
        assert state_path.exists()
        state = json.loads(state_path.read_text(encoding="utf-8"))
        assert state["read_only_rootfs"] is True
        assert state["network"] == "none"
        assert state["pids_limit"] == "256"
        assert state["memory_limit"] == "2g"
        assert state["cpu_limit"] == "2"
        assert state["cap_drops"] == ["ALL"]
        assert state["security_opts"] == ["no-new-privileges"]
        assert "/tmp:rw,noexec,nosuid,nodev,size=64m" in state["tmpfs_mounts"]
        assert "/run:rw,noexec,nosuid,nodev,size=16m" in state["tmpfs_mounts"]
        assert any(str(prepared.workspace_root) in mount for mount in state["bind_mounts"])
        assert any(str(prepared.artifact_root) in mount for mount in state["bind_mounts"])
        assert Path(prepared.metadata["docker_run_stdout_path"]).exists()
        assert Path(prepared.metadata["docker_run_stderr_path"]).exists()

        terminal_result = prepared.driver.run_terminal(
            "python3 -c \"print('alpha'); print(); print('beta')\""
        )
        verification_result = prepared.driver.run_verification(
            ("python3", "-c", "print('verified')"),
            label="verification",
        )
    finally:
        prepared.driver.close()

    assert terminal_result.timed_out is False
    assert "alpha\n\nbeta" in terminal_result.output
    assert verification_result.exit_code == 0
    assert state_path.exists() is False

    prepared.driver.close()


def test_local_workspace_backend_fetches_missing_prepared_ref(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    base, _ = _build_repo_with_second_commit(repo_root)
    prepared_ref = "refs/vtm-benchmark/example__repo-1/base"
    _run(repo_root, "git", "update-ref", prepared_ref, base)

    backend = LocalWorkspaceBackend()
    prepared = backend.prepare_workspace(
        case_id="prepared-ref-case",
        attempt_index=1,
        repo_root=repo_root,
        base_ref=prepared_ref,
        output_root=tmp_path / "outputs",
        mode="verified_lexical",
        command_timeout_seconds=30,
        max_output_chars=200,
    )

    try:
        assert _run(prepared.workspace_root, "git", "rev-parse", "HEAD") == base
    finally:
        prepared.driver.close()


def test_docker_workspace_backend_prepares_existing_workspace(
    tmp_path: Path,
    fake_docker_binary: Path,
) -> None:
    workspace_root = (tmp_path / "workspace").resolve()
    artifact_root = (tmp_path / "artifacts").resolve()
    workspace_root.mkdir(parents=True, exist_ok=True)
    (workspace_root / "hello.txt").write_text("hello\n", encoding="utf-8")
    backend = DockerWorkspaceBackend(
        docker_binary=str(fake_docker_binary),
        docker_image="python:3.12",
    )
    prepared = backend.prepare_existing_workspace(
        case_id="existing-shell-case",
        attempt_index=2,
        workspace_root=workspace_root,
        artifact_root=artifact_root,
        command_timeout_seconds=30,
        max_output_chars=200,
    )

    try:
        assert isinstance(prepared.driver, DockerWorkspaceDriver)
        assert prepared.workspace_root == workspace_root
        assert prepared.artifact_root == artifact_root
        result = prepared.driver.run_verification(
            (
                "python3",
                "-c",
                (
                    "from pathlib import Path; "
                    "print(Path('hello.txt').read_text(encoding='utf-8').strip())"
                ),
            ),
            label="verification",
        )
    finally:
        prepared.driver.close()

    assert result.exit_code == 0
    assert result.output == "hello"


def test_docker_workspace_backend_persists_startup_failure_logs(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    _build_repo(repo_root)
    failing_docker = tmp_path / "failing-docker"
    failing_docker.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "if len(sys.argv) > 1 and sys.argv[1] == 'run':\n"
        "    sys.stderr.write('container startup failed\\n')\n"
        "    raise SystemExit(42)\n"
        "raise SystemExit(0)\n",
        encoding="utf-8",
    )
    failing_docker.chmod(0o755)
    backend = DockerWorkspaceBackend(
        docker_binary=str(failing_docker),
        docker_image="python:3.12",
    )

    with pytest.raises(RuntimeError, match="docker workspace startup failed"):
        backend.prepare_workspace(
            case_id="shell-case",
            attempt_index=1,
            repo_root=repo_root,
            base_ref="HEAD",
            output_root=tmp_path / "outputs",
            mode="no_memory",
            command_timeout_seconds=30,
            max_output_chars=200,
        )

    artifact_root = (
        tmp_path / "outputs" / "executor-artifacts" / "shell-case" / "attempt-01"
    ).resolve()
    assert (artifact_root / "docker-run.stdout").read_text(encoding="utf-8") == ""
    assert (artifact_root / "docker-run.stderr").read_text(
        encoding="utf-8"
    ) == "container startup failed\n"
