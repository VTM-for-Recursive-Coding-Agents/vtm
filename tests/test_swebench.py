from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from vtm.benchmarks import BenchmarkManifest, BenchmarkRunConfig, BenchmarkRunner
from vtm.benchmarks.models import BenchmarkCaseResult, CodingTaskCase, CommitPair, RepoSpec
from vtm.benchmarks.repo_materialization import RepoWorkspaceManager
from vtm.benchmarks.swebench_harness import (
    SWEbenchHarnessInstanceResult,
    SWEbenchHarnessRunArtifacts,
    SWEbenchHarnessRunner,
)


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
                touched_paths=("smoke_module.py",),
                expected_changed_paths=("smoke_module.py",),
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
            mode="lexical",
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
        )
    ]
    results = [
        BenchmarkCaseResult(
            suite="coding",
            mode="lexical",
            case_id="example__repo-1",
            repo_name="example__repo",
            commit_pair_id="example__repo-1",
            metadata={"produced_patch_text": "diff --git a/foo b/foo\n"},
        )
    ]
    normalized, artifacts = runner.evaluate_predictions(
        cases=cases,
        results=results,
        config=BenchmarkRunConfig(
            manifest_path="swebench-harness.json",
            suite="coding",
            mode="lexical",
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
