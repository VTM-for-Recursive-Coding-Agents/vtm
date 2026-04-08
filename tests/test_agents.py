from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from vtm.adapters.git import GitRepoFingerprintCollector
from vtm.adapters.runtime import RuntimeEnvFingerprintCollector
from vtm.agents import (
    AgentConversationMessage,
    AgentMode,
    AgentModelTurnRequest,
    AgentModelTurnResponse,
    AgentRunRequest,
    AgentRunStatus,
    AgentRuntimeContext,
    AgentToolCall,
    DeterministicContextCompactor,
    InteractiveGuardedPermissionPolicy,
    TerminalCodingAgent,
)
from vtm.agents.tools import BuiltInToolProvider, ToolExecutionContext
from vtm.agents.workspace import LocalWorkspaceDriver
from vtm.enums import MemoryKind, ScopeKind, ValidityStatus
from vtm.memory_items import ClaimPayload, MemoryItem, ValidityState, VisibilityScope
from vtm.services import DependencyFingerprintBuilder


def _run(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        list(args),
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _build_bugfix_repo(repo: Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    _run(repo, "git", "init", "-b", "main")
    _run(repo, "git", "config", "user.name", "VTM Tests")
    _run(repo, "git", "config", "user.email", "vtm@example.com")
    (repo / "bugfix_module.py").write_text(
        "def buggy_increment(value: int) -> int:\n"
        '    """Return value plus one."""\n'
        "    return value\n",
        encoding="utf-8",
    )
    (repo / "tests").mkdir()
    (repo / "tests" / "test_bugfix_module.py").write_text(
        "import unittest\n\n"
        "from bugfix_module import buggy_increment\n\n\n"
        "class BugfixModuleTests(unittest.TestCase):\n"
        "    def test_buggy_increment(self) -> None:\n"
        "        self.assertEqual(buggy_increment(3), 4)\n\n\n"
        "if __name__ == '__main__':\n"
        "    unittest.main()\n",
        encoding="utf-8",
    )
    _run(repo, "git", "add", "bugfix_module.py", "tests/test_bugfix_module.py")
    _run(repo, "git", "commit", "-m", "base")


class FakeBugfixAgentModel:
    model_id = "fake-bugfix-agent"

    def __init__(self) -> None:
        self._step = 0
        self._recorded_memory_id: str | None = None

    def complete_turn(self, request: AgentModelTurnRequest) -> AgentModelTurnResponse:
        self._capture_memory_id(request)
        if self._step == 0:
            response = AgentModelTurnResponse(
                tool_calls=(
                    AgentToolCall(
                        tool_name="retrieve_memory",
                        arguments={"query": "buggy increment adds one", "limit": 3},
                    ),
                    AgentToolCall(
                        tool_name="read",
                        arguments={"path": "bugfix_module.py"},
                    ),
                )
            )
        elif self._step == 1:
            response = AgentModelTurnResponse(
                tool_calls=(
                    AgentToolCall(
                        tool_name="record_task_memory",
                        arguments={
                            "kind": "claim",
                            "title": "buggy_increment root cause",
                            "summary": "buggy_increment returns the input unchanged",
                            "claim": "buggy_increment returns the input unchanged",
                            "raw_content": "return value",
                            "tags": ["bugfix", "task-memory"],
                        },
                    ),
                )
            )
        elif self._step == 2:
            response = AgentModelTurnResponse(
                tool_calls=(
                    AgentToolCall(
                        tool_name="apply_patch",
                        arguments={"patch": self._bugfix_patch()},
                    ),
                )
            )
        elif self._step == 3:
            response = AgentModelTurnResponse(
                tool_calls=(
                    AgentToolCall(
                        tool_name="terminal",
                        arguments={
                            "command": " ".join(request.task_payload.get("test_command", []))
                        },
                    ),
                )
            )
        elif self._step == 4:
            if self._recorded_memory_id is None:
                raise AssertionError("expected record_task_memory result in prior messages")
            response = AgentModelTurnResponse(
                tool_calls=(
                    AgentToolCall(
                        tool_name="promote_procedure",
                        arguments={
                            "source_memory_ids": [self._recorded_memory_id],
                            "title": "Fix buggy_increment",
                            "summary": "Patch buggy_increment so it adds one.",
                            "goal": "Repair buggy_increment and verify with unittest.",
                            "steps": [
                                {
                                    "instruction": "Update buggy_increment to return value + 1.",
                                    "expected_outcome": "The function increments its input.",
                                },
                                {
                                    "instruction": "Run the targeted unittest file.",
                                    "expected_outcome": "The failing test passes.",
                                },
                            ],
                        },
                    ),
                )
            )
        else:
            response = AgentModelTurnResponse(assistant_message="task complete", done=True)
        self._step += 1
        return response

    def _capture_memory_id(self, request: AgentModelTurnRequest) -> None:
        for message in reversed(request.messages):
            if message.role != "tool" or message.tool_name != "record_task_memory":
                continue
            self._recorded_memory_id = str(json.loads(message.content)["memory_id"])
            return

    def _bugfix_patch(self) -> str:
        return (
            "diff --git a/bugfix_module.py b/bugfix_module.py\n"
            "index 8e8edcf..286dfbc 100644\n"
            "--- a/bugfix_module.py\n"
            "+++ b/bugfix_module.py\n"
            "@@ -1,3 +1,3 @@\n"
            " def buggy_increment(value: int) -> int:\n"
            "     \"\"\"Return value plus one.\"\"\"\n"
            "-    return value\n"
            "+    return value + 1\n"
        )


def _seed_repo_memory(kernel, repo_root: Path, durable_scope: VisibilityScope) -> None:
    dependency = DependencyFingerprintBuilder(
        repo_collector=GitRepoFingerprintCollector(),
        env_collector=RuntimeEnvFingerprintCollector(),
    ).build(
        str(repo_root),
        dependency_ids=("seed:bugfix",),
        input_digests=("bugfix_module.py",),
    )
    artifact = kernel.capture_artifact(
        b"buggy_increment should add one",
        content_type="text/plain",
        tool_name="seed",
    )
    seeded = MemoryItem(
        kind=MemoryKind.CLAIM,
        title="buggy_increment expectation",
        summary="buggy_increment should add one",
        payload=ClaimPayload(claim="buggy_increment should add one"),
        evidence=(kernel.artifact_evidence(artifact, label="seed"),),
        tags=("bugfix",),
        visibility=durable_scope,
        validity=ValidityState(
            status=ValidityStatus.VERIFIED,
            dependency_fingerprint=dependency,
        ),
    )
    tx = kernel.begin_transaction(durable_scope)
    kernel.stage_memory_item(tx.tx_id, seeded)
    kernel.commit_transaction(tx.tx_id)


def test_interactive_permission_policy_blocks_dangerous_terminal_command(tmp_path: Path) -> None:
    decision = InteractiveGuardedPermissionPolicy().authorize(
        tool_name="terminal",
        arguments={"command": "git reset --hard HEAD"},
        workspace_root=tmp_path,
    )

    assert decision.allowed is False
    assert "git reset --hard" in (decision.reason or "")


def test_deterministic_context_compactor_keeps_recent_messages() -> None:
    messages = tuple(
        AgentConversationMessage(role=role, content=f"message-{index}")
        for index, role in enumerate(
            ("system", "user", "assistant", "tool", "assistant", "tool"),
            start=1,
        )
    )

    compacted, record = DeterministicContextCompactor().compact(
        messages=messages,
        turn_index=3,
        window=4,
    )

    assert record is not None
    assert len(compacted) == 4
    assert compacted[0].role == "system"
    assert compacted[1].role == "user"
    assert "[Compacted context]" in compacted[2].content


def test_local_workspace_driver_enforces_timeout_and_recovers(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    _build_bugfix_repo(workspace)
    driver = LocalWorkspaceDriver(
        workspace_root=workspace,
        artifact_root=tmp_path / "artifacts",
        default_command_timeout_seconds=1,
        default_max_output_chars=200,
    )
    try:
        timed_out = driver.run_terminal(
            "python -c \"import time; time.sleep(2)\"",
            timeout_seconds=1,
        )
        recovered = driver.run_terminal("pwd")
    finally:
        driver.close()

    assert timed_out.timed_out is True
    assert timed_out.exit_code is None
    assert recovered.timed_out is False
    assert str(workspace) in recovered.output


def test_local_workspace_driver_truncates_terminal_output(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    _build_bugfix_repo(workspace)
    driver = LocalWorkspaceDriver(
        workspace_root=workspace,
        artifact_root=tmp_path / "artifacts",
        default_max_output_chars=40,
    )
    try:
        result = driver.run_terminal(
            "python -c \"print('x' * 200)\"",
            max_output_chars=40,
        )
    finally:
        driver.close()

    assert result.truncated is True
    assert len(result.output) == 40


class FakeWorkspaceDriver:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    def run_terminal(self, command: str, *, timeout_seconds=None, max_output_chars=None):
        self.calls.append(("terminal", command, timeout_seconds, max_output_chars))
        return _FakeCommandResult(
            operation="terminal",
            command=command,
            exit_code=0,
            output="terminal-output",
        )

    def read_file(self, path: str, *, start_line=1, end_line=None, max_chars=20000) -> str:
        self.calls.append(("read", path, start_line, end_line, max_chars))
        return "1: line"

    def search(self, pattern: str, *, path: str = "."):
        self.calls.append(("search", pattern, path))
        return _FakeCommandResult(
            operation="search",
            command=f"rg -n {pattern}",
            exit_code=0,
            output="bugfix_module.py:1:match",
        )

    def apply_patch(self, patch_text: str, *, patch_path=None):
        self.calls.append(("apply_patch", patch_text, str(patch_path)))
        return _FakeCommandResult(
            operation="apply_patch",
            command="git apply --3way",
            exit_code=0,
            output="patch applied",
        )

    def capture_patch(self) -> str:
        self.calls.append(("capture_patch",))
        return ""

    def capture_changed_paths(self) -> tuple[str, ...]:
        self.calls.append(("capture_changed_paths",))
        return ()

    def git_status(self) -> str:
        self.calls.append(("git_status",))
        return ""

    def run_verification(self, command: tuple[str, ...], *, label: str = "verification"):
        self.calls.append(("run_verification", command, label))
        return _FakeCommandResult(
            operation="verification",
            command=" ".join(command),
            exit_code=0,
            output="ok",
        )

    def close(self) -> None:
        self.calls.append(("close",))


class _FakeCommandResult:
    def __init__(
        self,
        *,
        operation: str,
        command: str,
        exit_code: int | None,
        output: str,
        stdout: str | None = None,
        stderr: str = "",
        duration_ms: float = 1.0,
        timed_out: bool = False,
        truncated: bool = False,
    ) -> None:
        self.operation = operation
        self.command = command
        self.exit_code = exit_code
        self.stdout = output if stdout is None else stdout
        self.stderr = stderr
        self.output = output
        self.duration_ms = duration_ms
        self.timed_out = timed_out
        self.truncated = truncated


def test_built_in_tools_delegate_workspace_operations(tmp_path: Path) -> None:
    driver = FakeWorkspaceDriver()
    context = ToolExecutionContext(
        workspace_root=tmp_path,
        task_file=tmp_path / "task.json",
        task_payload={"case_id": "fake"},
        artifact_root=tmp_path / "artifacts",
        workspace_driver=driver,
    )
    tools = BuiltInToolProvider().build_tools(context)

    read_result = tools["read"].execute({"path": "bugfix_module.py"}, context, "call-01")
    search_result = tools["search"].execute({"pattern": "buggy_increment"}, context, "call-02")
    patch_result = tools["apply_patch"].execute(
        {"patch": "diff --git a/a b/a\n--- a/a\n+++ b/a\n"},
        context,
        "call-03",
    )

    assert read_result.success is True
    assert search_result.success is True
    assert patch_result.success is True
    assert any(call[0] == "read" for call in driver.calls)
    assert any(call[0] == "search" for call in driver.calls)
    assert any(call[0] == "apply_patch" for call in driver.calls)


def test_terminal_coding_agent_writes_task_memory_and_promotes_procedure(
    tmp_path: Path,
    kernel,
    metadata_store,
) -> None:
    workspace = tmp_path / "workspace"
    _build_bugfix_repo(workspace)
    durable_scope = VisibilityScope(kind=ScopeKind.REPO, scope_id="agent-test-repo")
    _seed_repo_memory(kernel, workspace, durable_scope)

    task_payload = {
        "case_id": "bugfix",
        "task_statement": "Fix buggy_increment so it adds one.",
        "test_command": ["python", "-m", "unittest", "tests/test_bugfix_module.py"],
        "expected_changed_paths": ["bugfix_module.py"],
        "memory_context": [],
    }
    task_file = tmp_path / "task.json"
    task_file.write_text(json.dumps(task_payload, indent=2), encoding="utf-8")

    agent = TerminalCodingAgent(model_adapter=FakeBugfixAgentModel())
    driver = LocalWorkspaceDriver(
        workspace_root=workspace,
        artifact_root=tmp_path / "executor-artifacts" / "bugfix",
    )
    request = AgentRunRequest(
        session_id="agent-session-1",
        case_id="bugfix",
        task_file=str(task_file),
        workspace=str(workspace),
        model_id="fake-bugfix-agent",
        mode=AgentMode.BENCHMARK_AUTONOMOUS,
        task_payload=task_payload,
        max_turns=8,
        compaction_window=4,
    )
    context = AgentRuntimeContext(
        task_file=task_file,
        workspace_root=workspace,
        artifact_root=tmp_path / "agent-artifacts",
        task_payload=task_payload,
        test_command=("python", "-m", "unittest", "tests/test_bugfix_module.py"),
        workspace_driver=driver,
        kernel=kernel,
        task_scope=VisibilityScope(kind=ScopeKind.TASK, scope_id="agent-session-1"),
        durable_scope=durable_scope,
        dependency_builder=DependencyFingerprintBuilder(
            repo_collector=GitRepoFingerprintCollector(),
            env_collector=RuntimeEnvFingerprintCollector(),
        ),
    )

    try:
        result = agent.run(request, context)
    finally:
        driver.close()

    assert result.status is AgentRunStatus.COMPLETED
    assert result.turn_count >= 5
    assert result.terminal_command_count == 1
    assert result.compaction_count >= 1
    assert result.test_iterations == 1
    assert result.first_passing_turn is not None
    assert result.memory_write_count == 1
    assert result.memory_promotion_count == 1
    assert Path(result.artifacts["session"]).exists()
    assert Path(result.artifacts["turns_jsonl"]).exists()
    assert Path(result.artifacts["tool_calls_jsonl"]).exists()
    assert Path(result.artifacts["compactions_jsonl"]).exists()
    assert "return value + 1" in (workspace / "bugfix_module.py").read_text(encoding="utf-8")

    task_items = metadata_store.query_memory_items(
        scopes=(VisibilityScope(kind=ScopeKind.TASK, scope_id="agent-session-1"),)
    )
    durable_items = metadata_store.query_memory_items(scopes=(durable_scope,))

    assert len(task_items) == 1
    assert task_items[0].kind is MemoryKind.CLAIM
    assert any(item.kind is MemoryKind.PROCEDURE for item in durable_items)
    assert all(item.visibility.kind is ScopeKind.REPO for item in durable_items)
