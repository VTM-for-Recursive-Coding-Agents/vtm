from __future__ import annotations

import os
import subprocess
from pathlib import Path

import openai

from vtm.adapters.git import GitRepoFingerprintCollector
from vtm.adapters.runtime import RuntimeEnvFingerprintCollector
from vtm.enums import ScopeKind
from vtm.harness.executors import CodexBenchmarkExecutor, RLMBenchmarkExecutor
from vtm.harness.models import ExecutorRequest, HarnessTaskPack
from vtm.harness.workspace import LocalWorkspaceBackend
from vtm.memory_items import VisibilityScope
from vtm.retrieval import RetrieveRequest
from vtm.services import DependencyFingerprintBuilder
from vtm_rlm._vendored import vendored_rlm_root
from vtm_rlm.context import RLMRuntimeContext
from vtm_rlm.execution import VendoredRLMRunResult, run_vendored_rlm
from vtm_rlm.memory_bridge import VTMMemoryBridge, summarize_memory_context
from vtm_rlm.prompting import (
    CODING_RLM_SYSTEM_PROMPT,
    build_codex_task_prompt,
    build_phase1_task_prompt,
)


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
        memory_mode="lexical",
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
        memory_mode="lexical",
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


def test_build_codex_task_prompt_includes_verification_and_memory() -> None:
    task_pack = HarnessTaskPack(
        case_id="task-codex-1",
        repo_name="repo",
        commit_pair_id="pair",
        evaluation_backend="local_subprocess",
        base_ref="base",
        head_ref="head",
        task_statement="Fix the parser bug.",
        hints_text="Use the narrowest safe patch.",
        expected_changed_paths=("bug.py",),
        touched_paths=("bug.py",),
        test_command=("python", "-m", "pytest", "tests/test_bug.py", "-q"),
        target_patch_digest="deadbeef",
        memory_mode="lexical",
        top_k=5,
        memory_context=(
            {
                "memory_id": "m1",
                "title": "parser patch",
                "summary": "bug.py should return value + 1",
                "score": 0.9,
                "status": "verified",
            },
        ),
    )

    prompt = build_codex_task_prompt(task_pack)

    assert "Verification Command" in prompt
    assert "python -m pytest tests/test_bug.py -q" in prompt
    assert "Advisory VTM Memory" in prompt
    assert "Leave the repository with a valid patch" in prompt


def test_build_codex_task_prompt_filters_noisy_external_hints() -> None:
    task_pack = HarnessTaskPack(
        case_id="task-codex-2",
        repo_name="repo",
        commit_pair_id="pair",
        evaluation_backend="swebench_harness",
        base_ref="base",
        head_ref="head",
        task_statement="ascii.qdp Table format assumes QDP commands are upper case",
        hints_text=(
            "Welcome to Astropy and thanks for the issue!\n"
            "A project member will respond to you as soon as possible.\n"
            "The issue is that the regex that searches for QDP commands is not case "
            "insensitive.\n"
            "This attached patch fixes the issue, but there is probably a cleaner way.\n"
        ),
        fail_to_pass_tests=("astropy/io/ascii/tests/test_qdp.py::test_roundtrip[True]",),
        expected_changed_paths=("astropy/io/ascii/qdp.py",),
        touched_paths=("astropy/io/ascii/qdp.py",),
        localization_notes=("Failing test file: astropy/io/ascii/tests/test_qdp.py",),
        target_patch_digest="deadbeef",
        memory_mode="no_memory",
        top_k=5,
        task_kind="swebench_lite",
        difficulty="external",
    )

    prompt = build_codex_task_prompt(task_pack)

    assert "regex that searches for QDP commands is not case insensitive" in prompt
    assert "Welcome to Astropy" not in prompt
    assert "Expected Changed Paths" not in prompt
    assert "Failing test file: astropy/io/ascii/tests/test_qdp.py" in prompt


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


def test_run_vendored_rlm_sets_ollama_think_false(monkeypatch, tmp_path: Path) -> None:
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
        memory_mode="lexical",
        top_k=5,
        memory_context=(),
    )
    artifact_root = tmp_path / "artifacts"
    result = run_vendored_rlm(
        task_pack=task_pack,
        workspace_root=tmp_path,
        artifact_root=artifact_root,
        model_id="qwen3-coder:30b",
        kernel=None,
        scopes=(),
        max_iterations=2,
        max_depth=1,
        max_timeout_seconds=30,
        base_url="http://127.0.0.1:11434/v1",
        api_key="ollama",
    )

    assert result.response == "FINAL(ok)"
    assert captured["backend_kwargs"] == {
        "model_name": "qwen3-coder:30b",
        "base_url": "http://127.0.0.1:11434/v1",
        "completion_extra_body": {"think": False},
        "api_key": "ollama",
    }
    assert captured["custom_system_prompt"] == CODING_RLM_SYSTEM_PROMPT
    assert captured["root_prompt"] is None
    assert "PRELOADED_MEMORY" not in captured["custom_tools"]


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


def test_codex_executor_smoke_writes_patch(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    _build_bugfix_repo(repo_root)
    output_root = tmp_path / "out"
    prepared = LocalWorkspaceBackend().prepare_workspace(
        case_id="synthetic_bugfix_codex",
        attempt_index=1,
        repo_root=repo_root,
        base_ref="HEAD",
        output_root=output_root,
        mode="lexical",
        command_timeout_seconds=30,
        max_output_chars=4000,
    )
    task_pack = HarnessTaskPack(
        case_id="synthetic_bugfix_codex",
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
        memory_context=(
            {
                "memory_id": "m1",
                "title": "increment fix",
                "summary": "bugfix_module.py should return value + 1",
                "score": 0.99,
                "status": "verified",
            },
        ),
    )
    task_file = output_root / "task-codex.json"
    task_file.write_text(task_pack.model_dump_json(indent=2), encoding="utf-8")

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_codex = fake_bin / "codex"
    fake_codex.write_text(
        "#!/usr/bin/env python3\n"
        "import pathlib\n"
        "import sys\n"
        "\n"
        "args = sys.argv[1:]\n"
        "cwd = pathlib.Path.cwd()\n"
        "output = None\n"
        "index = 0\n"
        "while index < len(args):\n"
        "    if args[index] == '-C':\n"
        "        cwd = pathlib.Path(args[index + 1])\n"
        "        index += 2\n"
        "        continue\n"
        "    if args[index] == '-o':\n"
        "        output = pathlib.Path(args[index + 1])\n"
        "        index += 2\n"
        "        continue\n"
        "    index += 1\n"
        "prompt = sys.stdin.read()\n"
        "assert prompt\n"
        "(cwd / 'bugfix_module.py').write_text(\n"
        "    'def buggy_increment(value: int) -> int:\\n'\n"
        "    '    return value + 1\\n',\n"
        "    encoding='utf-8',\n"
        ")\n"
        "if output is not None:\n"
        "    output.write_text('Applied bugfix patch', encoding='utf-8')\n"
        "print('{\"event\":\"completed\"}')\n",
        encoding="utf-8",
    )
    fake_codex.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake_bin}:{os.environ['PATH']}")

    result = CodexBenchmarkExecutor(
        model_id="gpt-5.4",
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
        runtime_context=None,
    )

    assert result.command_exit_code == 0
    assert result.test_exit_code == 0
    assert "bugfix_module.py" in result.produced_patch_text
    assert result.agent_metadata["execution_engine"] == "codex"
    assert Path(result.agent_artifacts["codex_last_message_path"]).read_text(
        encoding="utf-8"
    ) == "Applied bugfix patch"


def test_codex_executor_timeout_preserves_partial_outputs(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    _build_bugfix_repo(repo_root)
    output_root = tmp_path / "out-timeout"
    prepared = LocalWorkspaceBackend().prepare_workspace(
        case_id="synthetic_bugfix_codex_timeout",
        attempt_index=1,
        repo_root=repo_root,
        base_ref="HEAD",
        output_root=output_root,
        mode="no_memory",
        command_timeout_seconds=30,
        max_output_chars=4000,
    )
    task_pack = HarnessTaskPack(
        case_id="synthetic_bugfix_codex_timeout",
        repo_name="synthetic_python_smoke",
        commit_pair_id="bugfix",
        evaluation_backend="local_subprocess",
        base_ref="HEAD",
        head_ref="HEAD",
        task_statement="Fix buggy_increment so the synthetic unit test passes again.",
        failing_tests=("tests.test_bugfix_module.BugfixModuleTests.test_buggy_increment",),
        target_patch_digest="deadbeef",
        memory_mode="no_memory",
        top_k=5,
    )
    task_file = output_root / "task-codex-timeout.json"
    task_file.write_text(task_pack.model_dump_json(indent=2), encoding="utf-8")

    original_run = subprocess.run

    def fake_run(*args, **kwargs):  # noqa: ANN002, ANN003
        command = args[0]
        if command[:2] != ["codex", "exec"]:
            return original_run(*args, **kwargs)
        workspace_root = Path(command[command.index("-C") + 1])
        (workspace_root / "bugfix_module.py").write_text(
            "def buggy_increment(value: int) -> int:\n"
            "    return value + 1\n",
            encoding="utf-8",
        )
        raise subprocess.TimeoutExpired(
            cmd=command,
            timeout=kwargs["timeout"],
            output=b'{"event":"partial"}\n',
            stderr=b"timed out waiting for codex\n",
        )

    monkeypatch.setattr("vtm.harness.executors.subprocess.run", fake_run)

    result = CodexBenchmarkExecutor(
        model_id="gpt-5.4",
        max_timeout_seconds=1,
    ).execute(
        request=ExecutorRequest(
            case_id=task_pack.case_id,
            task_file=str(task_file),
            workspace=str(prepared.workspace_root),
            artifact_root=str(prepared.artifact_root),
            attempt_index=1,
            workspace_backend=prepared.backend_name,
            test_command=(),
        ),
        prepared_workspace=prepared,
        runtime_context=None,
    )

    assert result.command_timed_out is True
    assert result.command_exit_code is None
    assert "bugfix_module.py" in result.produced_patch_text
    assert Path(result.command_stdout_path or "").read_text(encoding="utf-8") == (
        '{"event":"partial"}\n'
    )
    assert Path(result.command_stderr_path or "").read_text(encoding="utf-8") == (
        "timed out waiting for codex\n"
    )


def test_codex_executor_uses_verified_memory_as_second_phase_fallback(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_root = tmp_path / "repo"
    _build_bugfix_repo(repo_root)
    output_root = tmp_path / "out-codex-fallback"
    prepared = LocalWorkspaceBackend().prepare_workspace(
        case_id="synthetic_bugfix_codex_fallback",
        attempt_index=1,
        repo_root=repo_root,
        base_ref="HEAD",
        output_root=output_root,
        mode="verified_lexical",
        command_timeout_seconds=30,
        max_output_chars=4000,
    )
    task_pack = HarnessTaskPack(
        case_id="synthetic_bugfix_codex_fallback",
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
                "memory_id": "m1",
                "title": "increment fix",
                "summary": "bugfix_module.py should return value + 1",
                "score": 0.99,
                "status": "verified",
            },
            {
                "memory_id": "m2",
                "title": "stale hint",
                "summary": "ignore this stale note",
                "score": 0.50,
                "status": "stale",
            },
        ),
    )
    task_file = output_root / "task-codex-fallback.json"
    task_file.write_text(task_pack.model_dump_json(indent=2), encoding="utf-8")

    phase_prompts: list[str] = []
    original_run = subprocess.run

    def fake_run(*args, **kwargs):  # noqa: ANN002, ANN003
        command = args[0]
        if command[:2] != ["codex", "exec"]:
            return original_run(*args, **kwargs)
        prompt = kwargs["input"]
        phase_prompts.append(prompt)
        workspace_root = Path(command[command.index("-C") + 1])
        output_path = Path(command[command.index("-o") + 1])
        if len(phase_prompts) == 2 or "Advisory VTM Memory" in prompt:
            (workspace_root / "bugfix_module.py").write_text(
                "def buggy_increment(value: int) -> int:\n"
                "    return value + 1\n"
                "    # memory fallback\n",
                encoding="utf-8",
            )
            output_path.write_text("Applied memory-assisted patch", encoding="utf-8")
            return subprocess.CompletedProcess(
                command,
                0,
                stdout='{"event":"completed"}\n',
                stderr="",
            )
        output_path.write_text("Grounded on repo only", encoding="utf-8")
        return subprocess.CompletedProcess(
            command,
            0,
            stdout='{"event":"completed"}\n',
            stderr="",
        )

    monkeypatch.setattr("vtm.harness.executors.subprocess.run", fake_run)

    result = CodexBenchmarkExecutor(
        model_id="gpt-5.4",
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
        runtime_context=None,
    )

    assert result.test_exit_code == 0
    assert result.agent_metadata["codex_execution_strategy"] == "ground_then_verified_memory"
    assert len(result.agent_metadata["codex_phases"]) == 2
    assert "Advisory VTM Memory" not in phase_prompts[0]
    assert "Advisory VTM Memory" in phase_prompts[1]
    assert "ignore this stale note" not in phase_prompts[1]


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
        mode="lexical",
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
        memory_mode="lexical",
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
        else:
            (workspace_root / "bugfix_module.py").write_text(
                "def buggy_increment(value: int) -> int:\n"
                "return value + 1\n",
                encoding="utf-8",
            )
            response = "FINAL(I made a change.)"
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
            metadata={"model_id": model_id},
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
    phases = result.agent_metadata["rlm_phases"]
    assert phases[0]["corrective_retry_used"] is True
    assert phases[0]["corrective_retry_reason"] is not None
