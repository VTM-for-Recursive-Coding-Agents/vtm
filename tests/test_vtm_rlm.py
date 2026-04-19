from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from vtm.adapters.git import GitRepoFingerprintCollector
from vtm.adapters.runtime import RuntimeEnvFingerprintCollector
from vtm.enums import ScopeKind
from vtm.harness.executors import RLMBenchmarkExecutor
from vtm.harness.models import ExecutorRequest, HarnessTaskPack
from vtm.harness.workspace import LocalWorkspaceBackend
from vtm.memory_items import VisibilityScope
from vtm.retrieval import RetrieveRequest
from vtm.services import DependencyFingerprintBuilder
from vtm_rlm._vendored import ensure_vendored_rlm_on_path, vendored_rlm_root
from vtm_rlm.context import RLMRuntimeContext
from vtm_rlm.execution import VendoredRLMRunResult, run_vendored_rlm
from vtm_rlm.memory_bridge import VTMMemoryBridge, summarize_memory_context
from vtm_rlm.prompting import (
    CODING_RLM_SYSTEM_PROMPT,
    build_phase1_task_prompt,
)

openai = pytest.importorskip("openai")


def _extract_repl_blocks(text: str):
    ensure_vendored_rlm_on_path()
    from rlm.utils.parsing import extract_repl_blocks

    return extract_repl_blocks(text)


def _vendored_openai_client_class():
    ensure_vendored_rlm_on_path()
    from rlm.clients.openai import OpenAIClient

    return OpenAIClient


def _commit_memory(kernel, scope: VisibilityScope, memory) -> None:
    tx = kernel.begin_transaction(scope)
    kernel.stage_memory_item(tx.tx_id, memory)
    kernel.commit_transaction(tx.tx_id)


def test_vendored_rlm_root_exists() -> None:
    assert vendored_rlm_root().is_dir()


def test_extract_repl_blocks_keeps_fenced_repl_behavior() -> None:
    extraction = _extract_repl_blocks(
        "Plan\n```repl\nprint(context)\n```\nFINAL(done)"
    )

    assert extraction.code_blocks == ["print(context)"]
    assert extraction.fenced_repl_block_count == 1
    assert extraction.json_repl_block_count == 0


def test_extract_repl_blocks_supports_json_repl_object() -> None:
    extraction = _extract_repl_blocks('{"repl": {"code": "print(context)"}}')

    assert extraction.code_blocks == ["print(context)"]
    assert extraction.json_repl_block_count == 1


def test_extract_repl_blocks_supports_command_style_json_repl() -> None:
    extraction = _extract_repl_blocks(
        '{"command": "repl", "args": {"code": "print(context[:2000])"}}'
    )

    assert extraction.code_blocks == ["print(context[:2000])"]
    assert extraction.json_repl_block_count == 1


@pytest.mark.parametrize(
    "payload",
    (
        '{"command": "repl", "arguments": {"code": "print(context[:2000])"}}',
        '{"tool": "repl", "arguments": {"code": "print(context[:2000])"}}',
    ),
)
def test_extract_repl_blocks_supports_arguments_json_repl(payload: str) -> None:
    extraction = _extract_repl_blocks(payload)

    assert extraction.code_blocks == ["print(context[:2000])"]
    assert extraction.json_repl_block_count == 1


def test_extract_repl_blocks_supports_json_fence_wrapper() -> None:
    extraction = _extract_repl_blocks(
        '```json\n{"tool": "repl", "args": {"code": "print(context)"}}\n```'
    )

    assert extraction.code_blocks == ["print(context)"]
    assert extraction.json_repl_block_count == 1


def test_extract_repl_blocks_ignores_unrelated_json() -> None:
    extraction = _extract_repl_blocks('{"tool": "read_file", "args": {"path": "bug.py"}}')

    assert extraction.code_blocks == []
    assert extraction.json_repl_block_count == 0


def test_extract_repl_blocks_ignores_malformed_json_safely() -> None:
    extraction = _extract_repl_blocks("```json\n{\"repl\": {\"code\": print(context)}}\n```")

    assert extraction.code_blocks == []
    assert extraction.json_repl_block_count == 0


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
        memory_mode="verified_lexical",
        top_k=5,
        memory_context=(),
    )

    prompt = build_phase1_task_prompt(task_pack, Path("/tmp/workspace"))

    assert "Workspace root: /tmp/workspace" in prompt
    assert "Fix the bug." in prompt
    assert "search_memory" in prompt


def test_summarize_memory_context_marks_memory_as_advisory() -> None:
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
        memory_mode="verified_lexical",
        top_k=5,
        memory_context=(
            {
                "memory_id": "m1",
                "title": "bug should add one",
                "summary": "bug.py should return value + 1",
                "score": 0.9,
                "status": "verified",
            },
        ),
    )

    prompt = build_phase1_task_prompt(task_pack, Path("/tmp/workspace"))
    summary = summarize_memory_context(task_pack.memory_context)

    assert "Advisory VTM Memory" in prompt
    assert "trust the repository" in prompt
    assert "verify whether this is still true" in summary


def test_build_phase1_task_prompt_compacts_external_tasks() -> None:
    long_problem = (
        "ascii.qdp Table format assumes QDP commands are upper case.\n"
        "### Description\n\n"
        "A long issue body follows here that should not be injected into the first prompt.\n"
        "```python\n"
        "print('example')\n"
        "```"
    )
    task_pack = HarnessTaskPack(
        case_id="task-ext-1",
        repo_name="repo",
        commit_pair_id="pair",
        evaluation_backend="swebench_harness",
        base_ref="base",
        head_ref="head",
        task_statement=long_problem,
        problem_statement=long_problem,
        hints_text=(
            "The issue is that the regex that searches for QDP commands is not case "
            "insensitive.\n\nMore discussion follows."
        ),
        fail_to_pass_tests=("astropy/io/ascii/tests/test_qdp.py::test_roundtrip[True]",),
        pass_to_pass_tests=("one", "two"),
        expected_changed_paths=("astropy/io/ascii/qdp.py",),
        touched_paths=("astropy/io/ascii/qdp.py",),
        localization_notes=("Failing test file: astropy/io/ascii/tests/test_qdp.py",),
        target_patch_digest="deadbeef",
        memory_mode="verified_lexical",
        top_k=5,
        task_kind="swebench_lite",
        difficulty="external",
    )

    prompt = build_phase1_task_prompt(task_pack, Path("/tmp/workspace"))

    assert "Compact Task Policy" in prompt
    assert "ascii.qdp Table format assumes QDP commands are upper case." in prompt
    assert "astropy/io/ascii/tests/test_qdp.py::test_roundtrip[True]" in prompt
    assert "Localization Notes" in prompt
    assert "Failing test file: astropy/io/ascii/tests/test_qdp.py" in prompt
    assert "Expected Changed Paths" not in prompt
    assert "astropy/io/ascii/qdp.py" not in prompt
    assert "Pass-to-Pass Tests" not in prompt
    assert "```python" not in prompt
    assert (
        "The issue is that the regex that searches for QDP commands is not case "
        "insensitive."
    ) in prompt


def test_external_prompt_can_opt_into_expected_changed_paths_for_debug() -> None:
    task_pack = HarnessTaskPack(
        case_id="task-debug-1",
        repo_name="repo",
        commit_pair_id="pair",
        evaluation_backend="swebench_harness",
        base_ref="base",
        head_ref="head",
        task_statement="Fix the parser bug.",
        fail_to_pass_tests=("tests/test_bug.py::test_parser",),
        expected_changed_paths=("src/parser.py",),
        debug_expected_changed_paths=True,
        target_patch_digest="deadbeef",
        memory_mode="verified_lexical",
        top_k=5,
        task_kind="swebench_lite",
        difficulty="external",
    )

    prompt = build_phase1_task_prompt(task_pack, Path("/tmp/workspace"))

    assert "Expected Changed Paths" in prompt
    assert "src/parser.py" in prompt


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


class _FakeChatCompletionWithoutUsage:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


class _FakeCompletionsWithoutUsage:
    def create(self, *, model, messages, extra_body=None):  # noqa: ANN001, D401
        del model, messages, extra_body
        return _FakeChatCompletionWithoutUsage('{"repl": {"code": "print(context)"}}')


class _FakeOpenAIClientWithoutUsage:
    def __init__(self, **kwargs) -> None:  # noqa: ANN003
        self.base_url = kwargs.get("base_url")
        self.chat = type("_FakeChat", (), {"completions": _FakeCompletionsWithoutUsage()})()


def test_vendored_openai_client_allows_missing_usage(monkeypatch) -> None:
    monkeypatch.setattr(openai, "OpenAI", _FakeOpenAIClientWithoutUsage)
    monkeypatch.setattr(openai, "AsyncOpenAI", _FakeOpenAIClientWithoutUsage)

    client = _vendored_openai_client_class()(
        model_name="fake-model",
        base_url="https://openrouter.ai/api/v1",
        api_key="test-key",
    )

    response = client.completion("Fix the bug.")
    usage = client.get_usage_summary().to_dict()

    assert response == '{"repl": {"code": "print(context)"}}'
    assert usage["model_usage_summaries"]["fake-model"]["total_calls"] == 1
    assert usage["model_usage_summaries"]["fake-model"]["usage_missing"] is True
    assert client.get_last_usage().usage_missing is True


def test_run_vendored_rlm_passes_openai_compatible_backend_kwargs(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    class _FakeLogger:
        def __init__(self, *, log_dir: str) -> None:
            captured["log_dir"] = log_dir

    class _FakeCompletion:
        def __init__(self) -> None:
            self.response = "FINAL(ok)"
            self.execution_time = 0.1
            self.metadata = {}
            usage_payload = {"total_input_tokens": 1, "total_output_tokens": 1}
            self.usage_summary = type(
                "_UsageSummary",
                (),
                {"to_dict": staticmethod(lambda: usage_payload)},
            )()

        def to_dict(self) -> dict[str, object]:
            return {"response": self.response}

    class _FakeRLM:
        def __init__(self, **kwargs) -> None:  # noqa: ANN003
            captured["backend_kwargs"] = kwargs["backend_kwargs"]
            captured["custom_tools"] = kwargs["custom_tools"]
            captured["custom_system_prompt"] = kwargs["custom_system_prompt"]

        def completion(self, prompt: str, root_prompt: str | None = None) -> _FakeCompletion:
            captured["prompt"] = prompt
            captured["root_prompt"] = root_prompt
            return _FakeCompletion()

    monkeypatch.setattr("vtm_rlm.execution.load_rlm_runtime", lambda: (_FakeRLM, _FakeLogger))
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
        memory_mode="verified_lexical",
        top_k=5,
        memory_context=(),
    )
    artifact_root = tmp_path / "artifacts"
    result = run_vendored_rlm(
        task_pack=task_pack,
        workspace_root=tmp_path,
        artifact_root=artifact_root,
        model_id="google/gemma-4-31b-it:free",
        kernel=None,
        scopes=(),
        max_iterations=2,
        max_depth=1,
        max_timeout_seconds=30,
        base_url="https://openrouter.ai/api/v1",
        api_key="openrouter-test",
    )

    assert result.response == "FINAL(ok)"
    assert captured["backend_kwargs"] == {
        "model_name": "google/gemma-4-31b-it:free",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": "openrouter-test",
    }
    assert captured["custom_system_prompt"] == CODING_RLM_SYSTEM_PROMPT
    assert captured["root_prompt"] is None
    assert "PRELOADED_MEMORY" not in captured["custom_tools"]


def test_run_vendored_rlm_flags_tool_call_only_final_response(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class _FakeLogger:
        def __init__(self, *, log_dir: str) -> None:
            del log_dir

    class _FakeCompletion:
        def __init__(self) -> None:
            self.response = '{"command": "repl", "arguments": {"code": "print(context)"}}'
            self.execution_time = 0.1
            self.metadata = {}
            usage_payload = {"total_input_tokens": 1, "total_output_tokens": 1}
            self.usage_summary = type(
                "_UsageSummary",
                (),
                {"to_dict": staticmethod(lambda: usage_payload)},
            )()

        def to_dict(self) -> dict[str, object]:
            return {"response": self.response}

    class _FakeRLM:
        def __init__(self, **kwargs) -> None:  # noqa: ANN003
            del kwargs

        def completion(self, prompt: str, root_prompt: str | None = None) -> _FakeCompletion:
            del prompt, root_prompt
            return _FakeCompletion()

    monkeypatch.setattr("vtm_rlm.execution.load_rlm_runtime", lambda: (_FakeRLM, _FakeLogger))
    task_pack = HarnessTaskPack(
        case_id="task-1",
        repo_name="repo",
        commit_pair_id="pair",
        evaluation_backend="local_subprocess",
        base_ref="base",
        head_ref="head",
        task_statement="Fix the bug.",
        target_patch_digest="deadbeef",
        memory_mode="verified_lexical",
        top_k=5,
    )

    result = run_vendored_rlm(
        task_pack=task_pack,
        workspace_root=tmp_path,
        artifact_root=tmp_path / "artifacts",
        model_id="fake-model",
        kernel=None,
        scopes=(),
        max_iterations=2,
        max_depth=1,
        max_timeout_seconds=30,
    )

    assert result.metadata["vtm_execution_diagnostics"]["final_response_was_tool_call"] is True


def _build_bugfix_repo(repo: Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)

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
        mode="verified_lexical",
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
        memory_mode="verified_lexical",
        top_k=5,
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



def test_rlm_executor_uses_memory_as_second_phase_fallback(
    tmp_path: Path,
    monkeypatch,
    kernel,
) -> None:
    phase_calls: list[dict[str, object]] = []

    def fake_run_vendored_rlm(
        *,
        task_pack,
        workspace_root: Path,
        artifact_root: Path,
        model_id: str,
        kernel,
        scopes,
        max_iterations: int,
        max_depth: int,
        max_timeout_seconds: int,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> VendoredRLMRunResult:
        del scopes, max_iterations, max_depth, max_timeout_seconds, base_url, api_key
        artifact_root.mkdir(parents=True, exist_ok=True)
        phase_calls.append(
            {
                "memory_context_count": len(task_pack.memory_context),
                "kernel_enabled": kernel is not None,
                "artifact_root": str(artifact_root),
            }
        )
        if task_pack.memory_context:
            (workspace_root / "bugfix_module.py").write_text(
                "def buggy_increment(value: int) -> int:\n"
                "    return value + 1\n"
                "    # memory-assisted fix\n",
                encoding="utf-8",
            )
            response = "FINAL(Used verified memory after grounding to apply the fix.)"
        else:
            response = "FINAL(Grounded on the workspace first.)"
        response_path = artifact_root / "response.txt"
        response_path.write_text(response, encoding="utf-8")
        completion_json_path = artifact_root / "completion.json"
        completion_json_path.write_text('{"response": "ok"}', encoding="utf-8")
        metadata_json_path = artifact_root / "trajectory.json"
        metadata_json_path.write_text('{"mode": "fake"}', encoding="utf-8")
        return VendoredRLMRunResult(
            response=response,
            runtime_ms=25.0,
            response_path=str(response_path),
            completion_json_path=str(completion_json_path),
            metadata_json_path=str(metadata_json_path),
            trajectory_dir=str(artifact_root / "trajectory"),
            usage_summary={"total_input_tokens": 10, "total_output_tokens": 5, "total_cost": 0.0},
            metadata={"model_id": model_id},
        )

    monkeypatch.setattr("vtm.harness.executors.run_vendored_rlm", fake_run_vendored_rlm)

    repo_root = tmp_path / "repo"
    _build_bugfix_repo(repo_root)
    output_root = tmp_path / "out"
    prepared = LocalWorkspaceBackend().prepare_workspace(
        case_id="synthetic_bugfix_memory_fallback",
        attempt_index=1,
        repo_root=repo_root,
        base_ref="HEAD",
        output_root=output_root,
        mode="verified_lexical",
        command_timeout_seconds=30,
        max_output_chars=4000,
    )
    task_pack = HarnessTaskPack(
        case_id="synthetic_bugfix_memory_fallback",
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
        memory_mode="verified_lexical",
        top_k=5,
        memory_context=(
            {
                "memory_id": "mem-1",
                "title": "buggy_increment should add one",
                "summary": "buggy_increment should return value + 1",
                "score": 1.0,
                "status": "verified",
                "relative_path": "bugfix_module.py",
                "symbol": "buggy_increment",
            },
        ),
    )
    task_file = output_root / "task.json"
    task_file.write_text(task_pack.model_dump_json(indent=2), encoding="utf-8")

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
            attempt_index=1,
            workspace_backend=prepared.backend_name,
            test_command=task_pack.test_command,
        ),
        prepared_workspace=prepared,
        runtime_context=RLMRuntimeContext(
            kernel=kernel,
            task_scope=VisibilityScope(
                kind=ScopeKind.TASK,
                scope_id="synthetic_bugfix_memory_fallback",
            ),
            durable_scope=VisibilityScope(kind=ScopeKind.REPO, scope_id="synthetic_python_smoke"),
            dependency_builder=DependencyFingerprintBuilder(
                repo_collector=GitRepoFingerprintCollector(),
                env_collector=RuntimeEnvFingerprintCollector(),
            ),
        ),
    )

    assert result.test_exit_code == 0
    assert result.agent_metadata["rlm_execution_strategy"] == "ground_then_memory"
    assert len(phase_calls) == 3
    assert phase_calls[0]["memory_context_count"] == 0
    assert phase_calls[0]["kernel_enabled"] is False
    assert phase_calls[1]["memory_context_count"] == 0
    assert phase_calls[1]["kernel_enabled"] is False
    assert phase_calls[2]["memory_context_count"] == 1
    assert phase_calls[2]["kernel_enabled"] is True


def test_rlm_executor_uses_corrective_retry_for_failed_patch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    phase_calls: list[str] = []

    def fake_run_vendored_rlm(
        *,
        task_pack,
        workspace_root: Path,
        artifact_root: Path,
        model_id: str,
        kernel,
        scopes,
        max_iterations: int,
        max_depth: int,
        max_timeout_seconds: int,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> VendoredRLMRunResult:
        del kernel, scopes, max_iterations, max_depth, max_timeout_seconds, base_url, api_key
        artifact_root.mkdir(parents=True, exist_ok=True)
        phase_calls.append(task_pack.task_statement)
        if "Corrective Retry" in task_pack.task_statement:
            (workspace_root / "bugfix_module.py").write_text(
                "def buggy_increment(value: int) -> int:\n"
                "    return value + 1\n",
                encoding="utf-8",
            )
            response = "FINAL(Corrected the patch and fixed the unit test.)"
            diagnostics = {
                "rlm_iteration_count": 1,
                "rlm_executed_repl_block_count": 1,
                "rlm_detected_json_repl_count": 1,
                "rlm_final_response_had_json_repl": True,
                "tool_failure_count": 0,
                "usage_missing": False,
            }
        else:
            (workspace_root / "bugfix_module.py").write_text(
                "def buggy_increment(value: int) -> int:\n"
                "return value + 1\n",
                encoding="utf-8",
            )
            response = "FINAL(I made a change.)"
            diagnostics = {
                "rlm_iteration_count": 1,
                "rlm_executed_repl_block_count": 1,
                "rlm_detected_json_repl_count": 0,
                "rlm_final_response_had_json_repl": False,
                "tool_failure_count": 1,
                "usage_missing": True,
            }
        response_path = artifact_root / "response.txt"
        response_path.write_text(response, encoding="utf-8")
        completion_json_path = artifact_root / "completion.json"
        completion_json_path.write_text('{"response": "ok"}', encoding="utf-8")
        metadata_json_path = artifact_root / "trajectory.json"
        metadata_json_path.write_text(f'{{"model_id": "{model_id}"}}', encoding="utf-8")
        return VendoredRLMRunResult(
            response=response,
            runtime_ms=25.0,
            response_path=str(response_path),
            completion_json_path=str(completion_json_path),
            metadata_json_path=str(metadata_json_path),
            trajectory_dir=str(artifact_root / "trajectory"),
            usage_summary={"total_input_tokens": 10, "total_output_tokens": 5, "total_cost": 0.0},
            metadata={"model_id": model_id, "vtm_execution_diagnostics": diagnostics},
        )

    monkeypatch.setattr("vtm.harness.executors.run_vendored_rlm", fake_run_vendored_rlm)

    repo_root = tmp_path / "repo"
    _build_bugfix_repo(repo_root)
    output_root = tmp_path / "out"
    prepared = LocalWorkspaceBackend().prepare_workspace(
        case_id="synthetic_bugfix_corrective_retry",
        attempt_index=1,
        repo_root=repo_root,
        base_ref="HEAD",
        output_root=output_root,
        mode="no_memory",
        command_timeout_seconds=30,
        max_output_chars=4000,
    )
    task_pack = HarnessTaskPack(
        case_id="synthetic_bugfix_corrective_retry",
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
        memory_mode="no_memory",
        top_k=5,
    )
    task_file = output_root / "task.json"
    task_file.write_text(task_pack.model_dump_json(indent=2), encoding="utf-8")

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
            attempt_index=1,
            workspace_backend=prepared.backend_name,
            test_command=task_pack.test_command,
        ),
        prepared_workspace=prepared,
        runtime_context=RLMRuntimeContext(
            kernel=None,
            task_scope=VisibilityScope(
                kind=ScopeKind.TASK,
                scope_id="synthetic_bugfix_corrective_retry",
            ),
            durable_scope=None,
            dependency_builder=None,
        ),
    )

    assert result.test_exit_code == 0
    assert len(phase_calls) == 2
    assert "Corrective Retry" in phase_calls[1]
    assert "edit the repository" in phase_calls[1]
    assert "git diff" in phase_calls[1]
    phases = result.agent_metadata["rlm_phases"]
    assert phases[0]["corrective_retry_used"] is True
    assert phases[0]["corrective_retry_reason"] is not None
    assert result.agent_metadata["rlm_iteration_count"] == 2
    assert result.agent_metadata["rlm_executed_repl_block_count"] == 2
    assert result.agent_metadata["rlm_detected_json_repl_count"] == 1
    assert result.agent_metadata["rlm_final_response_had_json_repl"] is True
    assert result.agent_metadata["final_response_was_tool_call"] is False
    assert result.agent_metadata["rlm_usage_missing"] is True
    assert result.agent_metadata["rlm_tool_code_executed"] is True
    assert result.agent_metadata["corrective_retry_triggered"] is True
    assert result.agent_metadata["corrective_retry_had_nonempty_diff"] is True
    assert result.agent_metadata["rlm_corrective_retry_triggered"] is True
    assert result.agent_metrics["turn_count"] == 2
    assert result.agent_metrics["tool_call_count"] == 2
    assert result.agent_metrics["tool_failure_count"] == 1


def test_rlm_executor_records_empty_patch_diagnostics(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fake_run_vendored_rlm(
        *,
        task_pack,
        workspace_root: Path,
        artifact_root: Path,
        model_id: str,
        kernel,
        scopes,
        max_iterations: int,
        max_depth: int,
        max_timeout_seconds: int,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> VendoredRLMRunResult:
        del (
            task_pack,
            workspace_root,
            kernel,
            scopes,
            max_iterations,
            max_depth,
            max_timeout_seconds,
            base_url,
            api_key,
        )
        artifact_root.mkdir(parents=True, exist_ok=True)
        response_path = artifact_root / "response.txt"
        response_path.write_text("FINAL(No edit made.)", encoding="utf-8")
        completion_json_path = artifact_root / "completion.json"
        completion_json_path.write_text('{"response": "ok"}', encoding="utf-8")
        metadata_json_path = artifact_root / "trajectory.json"
        metadata_json_path.write_text(f'{{"model_id": "{model_id}"}}', encoding="utf-8")
        return VendoredRLMRunResult(
            response="FINAL(No edit made.)",
            runtime_ms=10.0,
            response_path=str(response_path),
            completion_json_path=str(completion_json_path),
            metadata_json_path=str(metadata_json_path),
            trajectory_dir=str(artifact_root / "trajectory"),
            usage_summary={"total_input_tokens": 0, "total_output_tokens": 0},
            metadata={
                "model_id": model_id,
                "vtm_execution_diagnostics": {
                    "rlm_iteration_count": 1,
                    "rlm_executed_repl_block_count": 1,
                    "rlm_detected_json_repl_count": 1,
                    "rlm_final_response_had_json_repl": True,
                    "tool_failure_count": 0,
                    "usage_missing": False,
                },
            },
        )

    monkeypatch.setattr("vtm.harness.executors.run_vendored_rlm", fake_run_vendored_rlm)

    repo_root = tmp_path / "repo"
    _build_bugfix_repo(repo_root)
    output_root = tmp_path / "out-empty-patch"
    prepared = LocalWorkspaceBackend().prepare_workspace(
        case_id="synthetic_bugfix_empty_patch",
        attempt_index=1,
        repo_root=repo_root,
        base_ref="HEAD",
        output_root=output_root,
        mode="no_memory",
        command_timeout_seconds=30,
        max_output_chars=4000,
    )
    task_pack = HarnessTaskPack(
        case_id="synthetic_bugfix_empty_patch",
        repo_name="synthetic_python_smoke",
        commit_pair_id="bugfix",
        evaluation_backend="swebench_harness",
        base_ref="HEAD",
        head_ref="HEAD",
        task_statement="Fix buggy_increment so the synthetic unit test passes again.",
        target_patch_digest="deadbeef",
        memory_mode="no_memory",
        top_k=5,
        task_kind="swebench_lite",
    )
    task_file = output_root / "task.json"
    task_file.write_text(task_pack.model_dump_json(indent=2), encoding="utf-8")

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
            attempt_index=1,
            workspace_backend=prepared.backend_name,
            test_command=task_pack.test_command,
        ),
        prepared_workspace=prepared,
        runtime_context=RLMRuntimeContext(
            kernel=None,
            task_scope=VisibilityScope(
                kind=ScopeKind.TASK,
                scope_id="synthetic_bugfix_empty_patch",
            ),
            durable_scope=None,
            dependency_builder=None,
        ),
    )

    assert result.produced_patch_text == ""
    assert result.agent_metadata["empty_patch"] is True
    assert result.agent_metadata["empty_patch_after_rlm_execution"] is True
    assert result.agent_metadata["rlm_tool_code_executed"] is True
    assert result.agent_metadata["corrective_retry_triggered"] is True
    assert result.agent_metadata["corrective_retry_had_nonempty_diff"] is False
    assert result.agent_metadata["rlm_corrective_retry_triggered"] is True
    assert result.agent_metadata["rlm_detected_json_repl_count"] == 2


def test_rlm_corrective_retry_hides_expected_changed_paths_for_external_tasks(
    tmp_path: Path,
    monkeypatch,
) -> None:
    retry_inputs: list[HarnessTaskPack] = []

    def fake_run_vendored_rlm(
        *,
        task_pack,
        workspace_root: Path,
        artifact_root: Path,
        model_id: str,
        kernel,
        scopes,
        max_iterations: int,
        max_depth: int,
        max_timeout_seconds: int,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> VendoredRLMRunResult:
        del kernel, scopes, max_iterations, max_depth, max_timeout_seconds, base_url, api_key
        artifact_root.mkdir(parents=True, exist_ok=True)
        retry_inputs.append(task_pack)
        if "Corrective Retry" in task_pack.task_statement:
            (workspace_root / "bugfix_module.py").write_text(
                "def buggy_increment(value: int) -> int:\n"
                "    return value + 1\n",
                encoding="utf-8",
            )
        else:
            (workspace_root / "bugfix_module.py").write_text(
                "def buggy_increment(value: int) -> int:\n"
                "return value + 1\n",
                encoding="utf-8",
            )
        response_path = artifact_root / "response.txt"
        response_path.write_text("FINAL(done)", encoding="utf-8")
        completion_json_path = artifact_root / "completion.json"
        completion_json_path.write_text('{"response": "ok"}', encoding="utf-8")
        metadata_json_path = artifact_root / "trajectory.json"
        metadata_json_path.write_text(f'{{"model_id": "{model_id}"}}', encoding="utf-8")
        return VendoredRLMRunResult(
            response="FINAL(done)",
            runtime_ms=25.0,
            response_path=str(response_path),
            completion_json_path=str(completion_json_path),
            metadata_json_path=str(metadata_json_path),
            trajectory_dir=str(artifact_root / "trajectory"),
            usage_summary={"total_input_tokens": 10, "total_output_tokens": 5, "total_cost": 0.0},
            metadata={"model_id": model_id},
        )

    monkeypatch.setattr("vtm.harness.executors.run_vendored_rlm", fake_run_vendored_rlm)

    repo_root = tmp_path / "repo"
    _build_bugfix_repo(repo_root)
    output_root = tmp_path / "out-external-retry"
    prepared = LocalWorkspaceBackend().prepare_workspace(
        case_id="synthetic_bugfix_external_retry",
        attempt_index=1,
        repo_root=repo_root,
        base_ref="HEAD",
        output_root=output_root,
        mode="no_memory",
        command_timeout_seconds=30,
        max_output_chars=4000,
    )
    task_pack = HarnessTaskPack(
        case_id="synthetic_bugfix_external_retry",
        repo_name="synthetic_python_smoke",
        commit_pair_id="bugfix",
        evaluation_backend="swebench_harness",
        base_ref="HEAD",
        head_ref="HEAD",
        task_statement="Fix buggy_increment so the synthetic unit test passes again.",
        hints_text="Start from the failing test.",
        fail_to_pass_tests=("tests/test_bugfix_module.py::BugfixModuleTests::test_buggy_increment",),
        expected_changed_paths=("bugfix_module.py",),
        touched_paths=("bugfix_module.py",),
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
        memory_mode="no_memory",
        top_k=5,
        task_kind="swebench_lite",
        difficulty="external",
    )
    task_file = output_root / "task.json"
    task_file.write_text(task_pack.model_dump_json(indent=2), encoding="utf-8")

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
            attempt_index=1,
            workspace_backend=prepared.backend_name,
            test_command=task_pack.test_command,
        ),
        prepared_workspace=prepared,
        runtime_context=RLMRuntimeContext(
            kernel=None,
            task_scope=VisibilityScope(
                kind=ScopeKind.TASK,
                scope_id="synthetic_bugfix_external_retry",
            ),
            durable_scope=None,
            dependency_builder=None,
        ),
    )

    assert result.test_exit_code == 0
    assert len(retry_inputs) == 2
    retry_task_pack = retry_inputs[1]
    assert "Corrective Retry" in retry_task_pack.task_statement
    assert "Focus on these files" not in retry_task_pack.task_statement
    assert "Expected Changed Paths" not in retry_task_pack.task_statement
    assert "Focus on these files: bugfix_module.py" not in retry_task_pack.task_statement
    assert "expected_changed_paths are scoring-only" in retry_task_pack.task_statement
    assert "Focus on these files" not in (retry_task_pack.hints_text or "")
    assert "Focus on these files: bugfix_module.py" not in (retry_task_pack.hints_text or "")
