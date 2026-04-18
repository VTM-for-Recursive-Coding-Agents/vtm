from __future__ import annotations

import builtins
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from vtm.benchmarks import BenchmarkManifest, BenchmarkRunConfig, BenchmarkRunner
from vtm.benchmarks.models import BenchmarkCaseResult, CodingTaskCase, CommitPair, RepoSpec
from vtm.benchmarks.repo_materialization import RepoWorkspaceCommandError, RepoWorkspaceManager
from vtm.benchmarks.swebench import (
    PreparedInstanceRefs,
    PreparedRepoCache,
    SWEbenchInstancePreparationError,
    SWEbenchLiteInstance,
    SWEbenchLitePreparer,
    SWEbenchPrepareCommandError,
)
from vtm.benchmarks.swebench_harness import (
    SWEbenchHarnessInstanceResult,
    SWEbenchHarnessRunArtifacts,
    SWEbenchHarnessRunner,
)
from vtm.harness.models import HarnessTaskPack


def _run(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        list(args),
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _build_swebench_repo(repo: Path) -> tuple[str, str]:
    repo.mkdir(parents=True, exist_ok=True)
    _run(repo, "git", "init", "-b", "main")
    _run(repo, "git", "config", "user.name", "VTM Tests")
    _run(repo, "git", "config", "user.email", "vtm@example.com")
    (repo / "smoke_module.py").write_text(
        "def buggy_increment(value: int) -> int:\n"
        "    return value\n",
        encoding="utf-8",
    )
    (repo / "tests").mkdir()
    (repo / "tests" / "test_smoke_module.py").write_text(
        "import unittest\n\n"
        "from smoke_module import buggy_increment\n\n\n"
        "class SmokeModuleTests(unittest.TestCase):\n"
        "    def test_buggy_increment(self) -> None:\n"
        "        self.assertEqual(buggy_increment(3), 4)\n\n\n"
        "if __name__ == '__main__':\n"
        "    unittest.main()\n",
        encoding="utf-8",
    )
    _run(repo, "git", "add", "smoke_module.py", "tests/test_smoke_module.py")
    _run(repo, "git", "commit", "-m", "base")
    base = _run(repo, "git", "rev-parse", "HEAD")
    (repo / "smoke_module.py").write_text(
        "def buggy_increment(value: int) -> int:\n"
        "    return value + 1\n",
        encoding="utf-8",
    )
    _run(repo, "git", "add", "smoke_module.py")
    _run(repo, "git", "commit", "-m", "fix")
    head = _run(repo, "git", "rev-parse", "HEAD")
    return base, head


def _git_diff(repo: Path, base: str, head: str, *paths: str) -> str:
    command = ["git", "diff", "--binary", "--no-ext-diff", f"{base}..{head}", "--", *paths]
    return _run(repo, *command)


def _make_swebench_instance(instance_id: str) -> SWEbenchLiteInstance:
    return SWEbenchLiteInstance(
        instance_id=instance_id,
        repo="example/repo",
        repo_name="example__repo",
        remote_url="https://github.com/example/repo.git",
        base_commit=f"base-{instance_id}",
        patch="diff --git a/example.py b/example.py\n",
        test_patch="",
        problem_statement=f"Fix {instance_id}.",
        hints_text=None,
        fail_to_pass_tests=(),
        pass_to_pass_tests=(),
        dataset_name="princeton-nlp/SWE-bench_Lite",
    )


def _make_swebench_row(instance_id: str) -> dict[str, object]:
    instance = _make_swebench_instance(instance_id)
    return {
        "instance_id": instance.instance_id,
        "repo": instance.repo,
        "remote_url": instance.remote_url,
        "base_commit": instance.base_commit,
        "patch": instance.patch,
        "test_patch": instance.test_patch,
        "problem_statement": instance.problem_statement,
        "FAIL_TO_PASS": [],
        "PASS_TO_PASS": [],
    }


def _make_prepare_error(
    instance: SWEbenchLiteInstance,
    *,
    patch_kind: str = "patch",
    stderr: str = "error: patch failed",
) -> SWEbenchInstancePreparationError:
    return SWEbenchInstancePreparationError(
        instance=instance,
        failure_stage="apply_patch",
        patch_kind=patch_kind,
        error=SWEbenchPrepareCommandError(
            command=("git", "apply", "--3way", "/tmp/failing.patch"),
            cwd=Path("/tmp/worktree"),
            returncode=128,
            stdout="",
            stderr=stderr,
        ),
    )


def test_prepare_swebench_manifest_cli_generates_local_refs(tmp_path: Path) -> None:
    remote_repo = tmp_path / "remote-repo"
    base, head = _build_swebench_repo(remote_repo)
    patch = _git_diff(remote_repo, base, head, "smoke_module.py")
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "instance_id": "example__repo-1",
                    "repo": "example/repo",
                    "remote_url": str(remote_repo),
                    "base_commit": base,
                    "patch": patch,
                    "test_patch": "",
                    "problem_statement": "Fix buggy_increment so it adds one.",
                        "FAIL_TO_PASS": [
                            "tests/test_smoke_module.py::SmokeModuleTests::test_buggy_increment"
                        ],
                    "PASS_TO_PASS": [],
                }
            ]
        ),
        encoding="utf-8",
    )
    output_manifest = tmp_path / "swebench-lite.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "vtm.benchmarks.prepare_swebench_lite",
            "--dataset-path",
            str(dataset_path),
            "--cache-root",
            str(tmp_path / "cache"),
            "--output-manifest",
            str(output_manifest),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    manifest = BenchmarkManifest.from_path(output_manifest)
    repo_root = Path(manifest.repos[0].remote_url or "")
    assert payload["manifest_id"] == "swebench_lite_generated"
    assert manifest.repos[0].repo_name == "example__repo"
    assert manifest.coding_tasks[0].evaluation_backend == "swebench_harness"
    assert manifest.coding_tasks[0].expected_changed_paths == ("smoke_module.py",)
    assert (
        _run(repo_root, "git", "rev-parse", "refs/vtm-swebench/example__repo-1/base") == base
    )
    gold_commit = _run(repo_root, "git", "rev-parse", "refs/vtm-swebench/example__repo-1/gold")
    assert gold_commit
    assert _git_diff(
        repo_root,
        "refs/vtm-swebench/example__repo-1/base",
        "refs/vtm-swebench/example__repo-1/gold",
        "smoke_module.py",
    )


def test_swebench_prepare_run_includes_stderr_in_failures(
    monkeypatch,
    tmp_path: Path,
) -> None:
    preparer = SWEbenchLitePreparer()

    def fake_run(command, cwd, check, capture_output, text):  # noqa: ANN001, ANN202
        del command, cwd, check, capture_output, text
        raise subprocess.CalledProcessError(
            returncode=128,
            cmd=["git", "apply", "/tmp/failing.patch"],
            output="stdout body",
            stderr="stderr body",
        )

    monkeypatch.setattr("vtm.benchmarks.swebench.subprocess.run", fake_run)

    with pytest.raises(SWEbenchPrepareCommandError) as exc_info:
        preparer._run(["git", "apply", "/tmp/failing.patch"], cwd=tmp_path)

    message = str(exc_info.value)
    assert "git apply /tmp/failing.patch" in message
    assert f"cwd: {tmp_path}" in message
    assert "return code: 128" in message
    assert "stdout body" in message
    assert "stderr body" in message


def test_swebench_load_rows_missing_datasets_package_reports_fix(
    monkeypatch,
) -> None:
    preparer = SWEbenchLitePreparer()
    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: ANN001, ANN202
        if name == "datasets":
            raise ModuleNotFoundError("No module named 'datasets'")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError) as exc_info:
        preparer._load_rows(
            dataset_name="princeton-nlp/SWE-bench_Lite",
            dataset_path=None,
        )

    message = str(exc_info.value)
    assert "uv sync --extra bench" in message
    assert "uv run --extra bench vtm-prepare-swebench-lite" in message


def test_prepare_instances_skip_failed_instance_and_write_report(
    monkeypatch,
    tmp_path: Path,
) -> None:
    preparer = SWEbenchLitePreparer()
    cache_root = tmp_path / "cache"
    repo_root = cache_root / "repos" / "example__repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    instances = [
        _make_swebench_instance("example__repo-1"),
        _make_swebench_instance("example__repo-2"),
    ]

    monkeypatch.setattr(
        preparer,
        "_prepare_repo_cache",
        lambda instance, repos_root: PreparedRepoCache(  # noqa: ARG005
            repo_root=repo_root,
            default_branch="main",
        ),
    )

    def fake_prepare_refs(*, instance, cache_root, repo_cache):  # noqa: ANN001, ANN202
        del cache_root, repo_cache
        if instance.instance_id == "example__repo-1":
            raise _make_prepare_error(instance, stderr="error: patch does not apply")
        return PreparedInstanceRefs(
            base_ref=f"refs/vtm-swebench/{instance.instance_id}/base",
            gold_ref=f"refs/vtm-swebench/{instance.instance_id}/gold",
        )

    monkeypatch.setattr(preparer, "_prepare_instance_refs", fake_prepare_refs)

    prepared = preparer.prepare_instances(
        instances=instances,
        cache_root=cache_root,
        skip_failed_instances=True,
    )

    assert [item.instance.instance_id for item in prepared] == ["example__repo-2"]
    report = json.loads((cache_root / "prepare-report.json").read_text(encoding="utf-8"))
    assert report["successful_instance_ids"] == ["example__repo-2"]
    assert report["skipped_instance_ids"] == ["example__repo-1"]
    skipped = report["skipped_instances"][0]
    assert skipped["instance_id"] == "example__repo-1"
    assert skipped["repo"] == "example/repo"
    assert skipped["failure_stage"] == "apply_patch"
    assert skipped["patch_kind"] == "patch"
    assert skipped["stderr"] == "error: patch does not apply"


def test_prepare_manifest_counts_successful_instances_when_skipping(
    monkeypatch,
    tmp_path: Path,
) -> None:
    preparer = SWEbenchLitePreparer()
    cache_root = tmp_path / "cache"
    output_manifest = tmp_path / "swebench-lite.json"
    repo_root = cache_root / "repos" / "example__repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    attempted: list[str] = []

    monkeypatch.setattr(
        preparer,
        "_load_rows",
        lambda dataset_name, dataset_path: [  # noqa: ARG005
            _make_swebench_row("example__repo-1"),
            _make_swebench_row("example__repo-2"),
            _make_swebench_row("example__repo-3"),
            _make_swebench_row("example__repo-4"),
            _make_swebench_row("example__repo-5"),
        ],
    )
    monkeypatch.setattr(
        preparer,
        "_prepare_repo_cache",
        lambda instance, repos_root: PreparedRepoCache(  # noqa: ARG005
            repo_root=repo_root,
            default_branch="main",
        ),
    )

    def fake_prepare_refs(*, instance, cache_root, repo_cache):  # noqa: ANN001, ANN202
        del cache_root, repo_cache
        attempted.append(instance.instance_id)
        if instance.instance_id in {"example__repo-1", "example__repo-2"}:
            raise _make_prepare_error(instance, stderr="error: patch failed")
        return PreparedInstanceRefs(
            base_ref=f"refs/vtm-swebench/{instance.instance_id}/base",
            gold_ref=f"refs/vtm-swebench/{instance.instance_id}/gold",
        )

    monkeypatch.setattr(preparer, "_prepare_instance_refs", fake_prepare_refs)

    manifest = preparer.prepare_manifest(
        dataset_name="princeton-nlp/SWE-bench_Lite",
        cache_root=cache_root,
        output_manifest=output_manifest,
        max_instances=3,
        skip_failed_instances=True,
    )

    assert attempted == [
        "example__repo-1",
        "example__repo-2",
        "example__repo-3",
        "example__repo-4",
        "example__repo-5",
    ]
    assert [case.case_id for case in manifest.coding_tasks] == [
        "example__repo-3",
        "example__repo-4",
        "example__repo-5",
    ]
    report = json.loads((cache_root / "prepare-report.json").read_text(encoding="utf-8"))
    assert report["successful_instance_ids"] == [
        "example__repo-3",
        "example__repo-4",
        "example__repo-5",
    ]
    assert report["skipped_instance_ids"] == ["example__repo-1", "example__repo-2"]


def test_repo_materialization_fetches_missing_prepared_ref_on_checkout(tmp_path: Path) -> None:
    remote_repo = tmp_path / "remote-repo"
    base, head = _build_swebench_repo(remote_repo)
    prepared_ref = "refs/vtm-swebench/example__repo-1/base"
    _run(remote_repo, "git", "update-ref", prepared_ref, base)

    repo_spec = RepoSpec(
        repo_name="example__repo",
        source_kind="git",
        remote_url=str(remote_repo),
        branch="main",
        commit_pairs=(
            CommitPair(
                pair_id="example__repo-1",
                base_ref=prepared_ref,
                head_ref=head,
            ),
        ),
    )
    manager = RepoWorkspaceManager()
    materialized = manager.materialize_repo(repo_spec, tmp_path / "corpus")

    missing_ref = subprocess.run(
        ["git", "rev-parse", prepared_ref],
        cwd=materialized,
        check=False,
        capture_output=True,
        text=True,
    )
    assert missing_ref.returncode != 0

    manager.git_checkout(materialized, prepared_ref)

    assert _run(materialized, "git", "rev-parse", "HEAD") == base


def test_repo_materialization_resolves_local_relative_remote_before_clone(
    tmp_path: Path,
    monkeypatch,
) -> None:
    remote_repo = tmp_path / ".benchmarks" / "swebench-lite" / "repos" / "example__repo"
    remote_repo.mkdir(parents=True)
    repo_spec = RepoSpec(
        repo_name="example__repo",
        source_kind="git",
        remote_url=".benchmarks/swebench-lite/repos/example__repo",
        branch="main",
        commit_pairs=(
            CommitPair(
                pair_id="example__repo-1",
                base_ref="base",
                head_ref="head",
            ),
        ),
    )
    manager = RepoWorkspaceManager()
    commands: list[tuple[list[str], Path | None, bool]] = []

    def fake_run(command, *, cwd=None, check=True):  # noqa: ANN001, ANN202
        commands.append((list(command), cwd, check))
        return subprocess.CompletedProcess(list(command), 0, stdout="", stderr="")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(manager, "run", fake_run)

    manager.materialize_repo(repo_spec, tmp_path / "corpus")

    assert commands
    assert commands[0][0][6] == str(remote_repo.resolve())


@pytest.mark.parametrize(
    ("remote_url", "expected_remote"),
    [
        ("https://github.com/example/repo.git", "https://github.com/example/repo.git"),
        ("git@github.com:example/repo.git", "git@github.com:example/repo.git"),
    ],
)
def test_repo_materialization_leaves_network_remotes_unchanged(
    tmp_path: Path,
    monkeypatch,
    remote_url: str,
    expected_remote: str,
) -> None:
    repo_spec = RepoSpec(
        repo_name="example__repo",
        source_kind="git",
        remote_url=remote_url,
        branch="main",
        commit_pairs=(
            CommitPair(
                pair_id="example__repo-1",
                base_ref="base",
                head_ref="head",
            ),
        ),
    )
    manager = RepoWorkspaceManager()
    commands: list[list[str]] = []

    def fake_run(command, *, cwd=None, check=True):  # noqa: ANN001, ANN202
        del cwd, check
        commands.append(list(command))
        return subprocess.CompletedProcess(list(command), 0, stdout="", stderr="")

    monkeypatch.setattr(manager, "run", fake_run)

    manager.materialize_repo(repo_spec, tmp_path / "corpus")

    assert commands[0][6] == expected_remote


def test_repo_materialization_clone_failure_includes_stderr(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_spec = RepoSpec(
        repo_name="example__repo",
        source_kind="git",
        remote_url="https://github.com/example/repo.git",
        branch="main",
        commit_pairs=(
            CommitPair(
                pair_id="example__repo-1",
                base_ref="base",
                head_ref="head",
            ),
        ),
    )
    manager = RepoWorkspaceManager()

    def fake_subprocess_run(command, cwd, check, capture_output, text):  # noqa: ANN001, ANN202
        del cwd, check, capture_output, text
        raise subprocess.CalledProcessError(
            returncode=128,
            cmd=list(command),
            output="clone stdout",
            stderr="fatal: repository not found",
        )

    monkeypatch.setattr("vtm.benchmarks.repo_materialization.subprocess.run", fake_subprocess_run)

    with pytest.raises(RepoWorkspaceCommandError) as exc_info:
        manager.materialize_repo(repo_spec, tmp_path / "corpus")

    message = str(exc_info.value)
    assert "git clone --quiet --filter=blob:none --branch main" in message
    assert "return code: 128" in message
    assert "clone stdout" in message
    assert "fatal: repository not found" in message


def test_repo_materialization_fails_for_existing_non_git_destination(tmp_path: Path) -> None:
    repo_spec = RepoSpec(
        repo_name="example__repo",
        source_kind="git",
        remote_url="https://github.com/example/repo.git",
        branch="main",
        commit_pairs=(
            CommitPair(
                pair_id="example__repo-1",
                base_ref="base",
                head_ref="head",
            ),
        ),
    )
    repo_root = tmp_path / "corpus" / "example__repo"
    repo_root.mkdir(parents=True)
    (repo_root / "README.txt").write_text("partial clone debris\n", encoding="utf-8")

    with pytest.raises(RuntimeError) as exc_info:
        RepoWorkspaceManager().materialize_repo(repo_spec, tmp_path / "corpus")

    message = str(exc_info.value)
    assert "existing repo root is not a complete git repository" in message
    assert str(repo_root) in message
    assert "delete the directory and rerun materialization" in message


def test_swebench_coding_suite_runs_fake_harness_and_writes_artifacts(
    tmp_path: Path,
    monkeypatch,
    install_fake_vendored_rlm,
) -> None:
    install_fake_vendored_rlm()
    remote_repo = tmp_path / "remote-harness"
    base, head = _build_swebench_repo(remote_repo)
    manifest = BenchmarkManifest(
        manifest_id="swebench_harness_manifest",
        repos=(
            RepoSpec(
                repo_name="example__repo",
                source_kind="git",
                remote_url=str(remote_repo),
                branch="main",
                commit_pairs=(
                    CommitPair(
                        pair_id="example__repo-1",
                        base_ref=base,
                        head_ref=head,
                    ),
                ),
            ),
        ),
        coding_tasks=(
            CodingTaskCase(
                case_id="example__repo-1",
                repo_name="example__repo",
                commit_pair_id="example__repo-1",
                evaluation_backend="swebench_harness",
                instance_id="example__repo-1",
                dataset_name="princeton-nlp/SWE-bench_Lite",
                task_statement="Fix buggy_increment so it adds one.",
                problem_statement="Fix buggy_increment so it adds one.",
                failing_tests=("tests/test_smoke_module.py::SmokeModuleTests::test_buggy_increment",),
                touched_paths=("oracle_only.py",),
                expected_changed_paths=("oracle_only.py",),
                task_kind="swebench_lite",
                difficulty="external",
            ),
        ),
    )
    class FakeHarnessRunner:
        def evaluate_predictions(self, *, cases, results, config, output_dir):
            from vtm.benchmarks.swebench_harness import SWEbenchHarnessRunner

            runner = SWEbenchHarnessRunner()
            predictions_path = runner.write_predictions(
                cases=cases,
                results=results,
                output_dir=output_dir,
                model_name_or_path="qwen-test",
            )
            logs_dir = output_dir / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            log_path = logs_dir / "example__repo-1.log"
            log_path.write_text("resolved", encoding="utf-8")
            results_path = output_dir / "swebench_harness_results.json"
            results_path.write_text(
                json.dumps(
                    {
                        "example__repo-1": {
                            "resolved": True,
                            "patch_applied": True,
                            "harness_status": "resolved",
                            "evaluation_log_path": str(log_path),
                        }
                    }
                ),
                encoding="utf-8",
            )
            return (
                {
                    "example__repo-1": SWEbenchHarnessInstanceResult(
                        instance_id="example__repo-1",
                        resolved=True,
                        patch_applied=True,
                        harness_status="resolved",
                        evaluation_log_path=str(log_path),
                    )
                },
                SWEbenchHarnessRunArtifacts(
                    predictions_path=str(predictions_path),
                    results_path=str(results_path),
                    logs_dir=str(logs_dir),
                    stdout_path=str(output_dir / "harness.stdout"),
                    stderr_path=str(output_dir / "harness.stderr"),
                ),
            )

    monkeypatch.setattr(
        "vtm.benchmarks.suite_execution.BenchmarkSuiteExecutor._swebench_harness_runner",
        lambda self, output_dir: FakeHarnessRunner(),
    )

    result = BenchmarkRunner(
        manifest,
        BenchmarkRunConfig(
            manifest_path="swebench-harness.json",
            suite="coding",
            mode="verified_lexical",
            output_dir=str(tmp_path / "swebench-run"),
            rlm_model_id="fake-model",
            swebench_dataset_name="princeton-nlp/SWE-bench_Lite",
        ),
    ).run()

    result_rows = Path(result.artifacts["results_jsonl"]).read_text(encoding="utf-8").splitlines()
    row = json.loads(result_rows[0])
    assert result.case_count == 1
    assert result.metrics["resolved_count"] == 1
    assert result.metrics["pass_rate"] == 1.0
    assert result.metrics["resolved_rate"] == 1.0
    assert result.metrics["patch_applied_rate"] == 1.0
    assert row["metrics"]["incomplete"] is False
    assert row["metrics"]["resolved"] is True
    assert row["metadata"]["harness_status"] == "resolved"
    assert "predictions_jsonl" in result.artifacts
    assert "swebench_harness_results_json" in result.artifacts


def test_swebench_harness_uses_absolute_predictions_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = SWEbenchHarnessRunner()
    captured: dict[str, object] = {}

    def fake_run(command, cwd, check, capture_output, text):
        del check, capture_output, text
        captured["command"] = list(command)
        captured["cwd"] = cwd
        report_path = Path(cwd) / "fake-report.json"
        report_path.write_text(
            json.dumps(
                {
                    "resolved_ids": ["example__repo-1"],
                    "unresolved_ids": [],
                    "applied_ids": ["example__repo-1"],
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("vtm.benchmarks.swebench_harness.subprocess.run", fake_run)

    cases = [
        CodingTaskCase(
            case_id="example__repo-1",
            repo_name="example__repo",
            commit_pair_id="example__repo-1",
            evaluation_backend="swebench_harness",
            instance_id="example__repo-1",
            dataset_name="princeton-nlp/SWE-bench_Lite",
            task_statement="Fix buggy_increment so it adds one.",
            problem_statement="Fix buggy_increment so it adds one.",
        ),
        CodingTaskCase(
            case_id="example__repo-2",
            repo_name="example__repo",
            commit_pair_id="example__repo-2",
            evaluation_backend="swebench_harness",
            instance_id="example__repo-2",
            dataset_name="princeton-nlp/SWE-bench_Lite",
            task_statement="Fix another bug.",
            problem_statement="Fix another bug.",
        ),
    ]
    results = [
        BenchmarkCaseResult(
            suite="coding",
            mode="verified_lexical",
            case_id="example__repo-1",
            repo_name="example__repo",
            commit_pair_id="example__repo-1",
            metadata={"produced_patch_text": "diff --git a/foo b/foo\n"},
        ),
        BenchmarkCaseResult(
            suite="coding",
            mode="verified_lexical",
            case_id="example__repo-2",
            repo_name="example__repo",
            commit_pair_id="example__repo-2",
            metadata={"produced_patch_text": "diff --git a/bar b/bar\n"},
        ),
    ]
    normalized, artifacts = runner.evaluate_predictions(
        cases=cases,
        results=results,
        config=BenchmarkRunConfig(
            manifest_path="swebench-harness.json",
            suite="coding",
            mode="verified_lexical",
            output_dir=str(tmp_path / "run"),
            swebench_dataset_name="princeton-nlp/SWE-bench_Lite",
            swebench_harness_workers=1,
        ),
        output_dir=tmp_path / "run",
    )
    assert normalized["example__repo-1"].resolved is True
    command = captured["command"]
    assert isinstance(command, list)
    predictions_path = command[command.index("--predictions_path") + 1]
    assert Path(predictions_path).is_absolute()
    assert artifacts.predictions_path == predictions_path
    instance_ids_index = command.index("--instance_ids")
    assert command[instance_ids_index + 1 : instance_ids_index + 3] == [
        "example__repo-1",
        "example__repo-2",
    ]


def test_swebench_harness_failure_reports_docker_unavailable(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = SWEbenchHarnessRunner()

    def fake_run(command, cwd, check, capture_output, text):  # noqa: ANN001, ANN202
        del command, cwd, check, capture_output, text
        return SimpleNamespace(
            returncode=1,
            stdout="",
            stderr=(
                "Traceback (most recent call last):\n"
                "docker.errors.DockerException: Error while fetching server API version: "
                "('Connection aborted.', FileNotFoundError(2, 'No such file or directory'))\n"
            ),
        )

    monkeypatch.setattr("vtm.benchmarks.swebench_harness.subprocess.run", fake_run)

    cases = [
        CodingTaskCase(
            case_id="example__repo-1",
            repo_name="example__repo",
            commit_pair_id="example__repo-1",
            evaluation_backend="swebench_harness",
            instance_id="example__repo-1",
            dataset_name="princeton-nlp/SWE-bench_Lite",
            task_statement="Fix buggy_increment so it adds one.",
            problem_statement="Fix buggy_increment so it adds one.",
        )
    ]
    results = [
        BenchmarkCaseResult(
            suite="coding",
            mode="verified_lexical",
            case_id="example__repo-1",
            repo_name="example__repo",
            commit_pair_id="example__repo-1",
            metadata={"produced_patch_text": "diff --git a/foo b/foo\n"},
        )
    ]

    with pytest.raises(RuntimeError) as exc_info:
        runner.evaluate_predictions(
            cases=cases,
            results=results,
            config=BenchmarkRunConfig(
                manifest_path="swebench-harness.json",
                suite="coding",
                mode="verified_lexical",
                output_dir=str(tmp_path / "run"),
                swebench_dataset_name="princeton-nlp/SWE-bench_Lite",
                swebench_harness_workers=1,
            ),
            output_dir=tmp_path / "run",
        )

    message = str(exc_info.value)
    assert "Docker daemon is unavailable" in message
    assert "harness.stderr" in message


def test_external_task_pack_avoids_changed_path_leakage_in_retrieval_query(
    tmp_path: Path,
    monkeypatch,
    install_fake_vendored_rlm,
) -> None:
    install_fake_vendored_rlm()
    remote_repo = tmp_path / "remote-visible-query"
    base, head = _build_swebench_repo(remote_repo)
    manifest = BenchmarkManifest(
        manifest_id="swebench_query_manifest",
        repos=(
            RepoSpec(
                repo_name="example__repo",
                source_kind="git",
                remote_url=str(remote_repo),
                branch="main",
                commit_pairs=(
                    CommitPair(
                        pair_id="example__repo-1",
                        base_ref=base,
                        head_ref=head,
                    ),
                ),
            ),
        ),
        coding_tasks=(
            CodingTaskCase(
                case_id="example__repo-1",
                repo_name="example__repo",
                commit_pair_id="example__repo-1",
                evaluation_backend="swebench_harness",
                instance_id="example__repo-1",
                dataset_name="princeton-nlp/SWE-bench_Lite",
                task_statement="Fix buggy_increment so it adds one.",
                problem_statement="Fix buggy_increment so it adds one.",
                failing_tests=("tests/test_smoke_module.py::SmokeModuleTests::test_buggy_increment",),
                touched_paths=("oracle_only.py",),
                expected_changed_paths=("oracle_only.py",),
                task_kind="swebench_lite",
                difficulty="external",
            ),
        ),
    )

    class FakeHarnessRunner:
        def evaluate_predictions(self, *, cases, results, config, output_dir):
            del cases, results, config
            logs_dir = output_dir / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            log_path = logs_dir / "example__repo-1.log"
            log_path.write_text("resolved", encoding="utf-8")
            results_path = output_dir / "swebench_harness_results.json"
            results_path.write_text(
                json.dumps(
                    {
                        "example__repo-1": {
                            "resolved": True,
                            "patch_applied": True,
                            "harness_status": "resolved",
                            "evaluation_log_path": str(log_path),
                        }
                    }
                ),
                encoding="utf-8",
            )
            predictions_path = output_dir / "predictions.jsonl"
            predictions_path.write_text("", encoding="utf-8")
            return (
                {
                    "example__repo-1": SWEbenchHarnessInstanceResult(
                        instance_id="example__repo-1",
                        resolved=True,
                        patch_applied=True,
                        harness_status="resolved",
                        evaluation_log_path=str(log_path),
                    )
                },
                SWEbenchHarnessRunArtifacts(
                    predictions_path=str(predictions_path),
                    results_path=str(results_path),
                    logs_dir=str(logs_dir),
                    stdout_path=str(output_dir / "harness.stdout"),
                    stderr_path=str(output_dir / "harness.stderr"),
                ),
            )

    monkeypatch.setattr(
        "vtm.benchmarks.suite_execution.BenchmarkSuiteExecutor._swebench_harness_runner",
        lambda self, output_dir: FakeHarnessRunner(),
    )

    BenchmarkRunner(
        manifest,
        BenchmarkRunConfig(
            manifest_path="swebench-query.json",
            suite="coding",
            mode="verified_lexical",
            output_dir=str(tmp_path / "swebench-query-run"),
            rlm_model_id="fake-model",
            swebench_dataset_name="princeton-nlp/SWE-bench_Lite",
        ),
    ).run()

    task_pack = HarnessTaskPack.from_json(
        (
            tmp_path / "swebench-query-run" / "task-packs" / "example__repo-1.json"
        ).read_text(encoding="utf-8")
    )

    assert task_pack.expected_changed_paths == ("oracle_only.py",)
    assert task_pack.retrieval_query is not None
    assert "oracle_only.py" not in task_pack.retrieval_query
    assert "tests/test_smoke_module.py" in task_pack.retrieval_query
    assert task_pack.localization_notes == (
        "Failing test file: tests/test_smoke_module.py",
        "Referenced module from test import: smoke_module.py",
    )
