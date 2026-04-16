from __future__ import annotations

from pathlib import Path

import openai

from vtm.adapters.git import GitRepoFingerprintCollector
from vtm.adapters.runtime import RuntimeEnvFingerprintCollector
from vtm.enums import ScopeKind
from vtm.harness.executors import RLMBenchmarkExecutor
from vtm.harness.models import ExecutorRequest, HarnessTaskPack
from vtm.harness.workspace import LocalWorkspaceBackend
from vtm.memory_items import VisibilityScope
from vtm.retrieval import RetrieveRequest
from vtm.services import DependencyFingerprintBuilder
from vtm_rlm._vendored import vendored_rlm_root
from vtm_rlm.context import RLMRuntimeContext
from vtm_rlm.memory_bridge import VTMMemoryBridge
from vtm_rlm.prompting import build_phase1_task_prompt


def _commit_memory(kernel, scope: VisibilityScope, memory) -> None:
    tx = kernel.begin_transaction(scope)
    kernel.stage_memory_item(tx.tx_id, memory)
    kernel.commit_transaction(tx.tx_id)


def test_vendored_rlm_root_exists() -> None:
    assert vendored_rlm_root().is_dir()


def test_memory_bridge_search_and_expand(kernel, memory_factory, scope: VisibilityScope) -> None:
    memory = memory_factory(title="Parser fallback", summary="Parser fallback logic")
    _commit_memory(kernel, scope, memory)

    bridge = VTMMemoryBridge(kernel=kernel, scopes=(scope,))
    results = bridge.search_memory("fallback parser", limit=3)

    assert results
    assert results[0]["memory_id"] == memory.memory_id

    evidence = bridge.expand_memory(memory.memory_id)
    assert evidence
    assert evidence[0]["kind"] == "artifact"


def test_build_phase1_task_prompt_includes_memory_context() -> None:
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
        target_patch_digest="deadbeef",
        memory_mode="lexical",
        top_k=5,
        coding_executor="rlm",
        memory_context=(),
    )

    prompt = build_phase1_task_prompt(task_pack, Path("/tmp/workspace"))

    assert "Workspace root: /tmp/workspace" in prompt
    assert "Fix the bug." in prompt
    assert "search_memory" in prompt


class _FakeUsage:
    prompt_tokens = 10
    completion_tokens = 20
    total_tokens = 30


class _FakeChoiceMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeChoiceMessage(content)


class _FakeChatCompletion:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]
        self.usage = _FakeUsage()


class _FakeCompletions:
    def create(self, *, model, messages, extra_body=None):  # noqa: ANN001, D401
        del model, messages, extra_body
        return _FakeChatCompletion(
            "\n".join(
                [
                    "Applying a direct workspace fix.",
                    "```repl",
                    "from pathlib import Path",
                    "Path(WORKSPACE_ROOT, 'bugfix_module.py').write_text(",
                    "    'def buggy_increment(value: int) -> int:\\n'"
                    "    '    return value + 1\\n',",
                    "    encoding='utf-8',",
                    ")",
                    "```",
                    "FINAL(Fixed buggy_increment and validated the unit test.)",
                ]
            )
        )


class _FakeChat:
    def __init__(self) -> None:
        self.completions = _FakeCompletions()


class _FakeOpenAIClient:
    def __init__(self, **kwargs) -> None:  # noqa: ANN003
        self.base_url = kwargs.get("base_url")
        self.chat = _FakeChat()


def _build_bugfix_repo(repo: Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    import subprocess

    subprocess.run(["git", "init", "-b", "main"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.name", "VTM RLM Tests"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "vtm-rlm@example.com"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    (repo / "bugfix_module.py").write_text(
        "def buggy_increment(value: int) -> int:\n"
        "    return value - 1\n",
        encoding="utf-8",
    )
    tests_dir = repo / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    (tests_dir / "test_bugfix_module.py").write_text(
        "import unittest\n"
        "from bugfix_module import buggy_increment\n\n"
        "class BugfixModuleTests(unittest.TestCase):\n"
        "    def test_buggy_increment(self) -> None:\n"
        "        self.assertEqual(buggy_increment(2), 3)\n\n"
        "if __name__ == '__main__':\n"
        "    unittest.main()\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=repo, check=True, capture_output=True)


def test_rlm_executor_smoke_writes_patch_and_memory(
    tmp_path: Path,
    monkeypatch,
    kernel,
    metadata_store,
) -> None:
    monkeypatch.setattr(openai, "OpenAI", _FakeOpenAIClient)
    monkeypatch.setattr(openai, "AsyncOpenAI", _FakeOpenAIClient)

    repo_root = tmp_path / "repo"
    _build_bugfix_repo(repo_root)
    output_root = tmp_path / "out"
    prepared = LocalWorkspaceBackend().prepare_workspace(
        case_id="synthetic_bugfix_unittest",
        attempt_index=1,
        repo_root=repo_root,
        base_ref="HEAD",
        output_root=output_root,
        mode="lexical",
        command_timeout_seconds=30,
        max_output_chars=4000,
    )
    task_pack = HarnessTaskPack(
        case_id="synthetic_bugfix_unittest",
        repo_name="synthetic_python_smoke",
        commit_pair_id="bugfix",
        evaluation_backend="local_subprocess",
        base_ref="HEAD",
        head_ref="HEAD",
        task_statement="Fix buggy_increment so the synthetic unit test passes again.",
        failing_tests=("tests.test_bugfix_module.BugfixModuleTests.test_buggy_increment",),
        test_command=(
            "python3",
            "-m",
            "unittest",
            "discover",
            "-s",
            "tests",
            "-p",
            "test_*.py",
            "-q",
        ),
        target_patch_digest="deadbeef",
        memory_mode="lexical",
        top_k=5,
        coding_executor="rlm",
    )
    task_file = output_root / "task.json"
    task_file.write_text(task_pack.model_dump_json(indent=2), encoding="utf-8")
    durable_scope = VisibilityScope(kind=ScopeKind.REPO, scope_id="synthetic_python_smoke")

    result = RLMBenchmarkExecutor(
        model_id="fake-model",
        max_iterations=4,
        max_timeout_seconds=30,
    ).execute(
        request=ExecutorRequest(
            case_id=task_pack.case_id,
            task_file=str(task_file),
            workspace=str(prepared.workspace_root),
            artifact_root=str(prepared.artifact_root),
            coding_executor="rlm",
            attempt_index=1,
            workspace_backend=prepared.backend_name,
            test_command=task_pack.test_command,
        ),
        prepared_workspace=prepared,
        runtime_context=RLMRuntimeContext(
            kernel=kernel,
            task_scope=VisibilityScope(kind=ScopeKind.TASK, scope_id="synthetic_bugfix_unittest"),
            durable_scope=durable_scope,
            dependency_builder=DependencyFingerprintBuilder(
                repo_collector=GitRepoFingerprintCollector(),
                env_collector=RuntimeEnvFingerprintCollector(),
            ),
        ),
    )

    assert result.command_exit_code == 0
    assert result.test_exit_code == 0
    assert "return value + 1" in result.produced_patch_text
    assert result.agent_metadata["rlm_memory_id"] is not None
    assert Path(result.agent_artifacts["rlm_completion_json_path"]).exists()

    retrieval = kernel.retrieve(
        RetrieveRequest(
            query="vendored rlm result bugfix",
            scopes=(durable_scope,),
            limit=5,
        )
    )
    assert retrieval.candidates
    assert any(
        candidate.memory.metadata.get("generated_by") == "vendored_rlm"
        for candidate in retrieval.candidates
    )
