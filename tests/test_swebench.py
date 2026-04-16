from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

from vtm.benchmarks import BenchmarkManifest, BenchmarkRunConfig, BenchmarkRunner
from vtm.benchmarks.local_patcher import LocalOpenAIPatcher, LocalPatcherConfig
from vtm.benchmarks.models import BenchmarkCaseResult, CodingTaskCase, CommitPair, RepoSpec
from vtm.benchmarks.repo_materialization import RepoWorkspaceManager
from vtm.benchmarks.scaffold_bridge import ScaffoldBridge
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


def _start_fake_openai_server(response_text: str) -> tuple[ThreadingHTTPServer, str]:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers["Content-Length"])
            self.server.requests.append(self.rfile.read(length).decode("utf-8"))  # type: ignore[attr-defined]
            body = json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": response_text,
                            }
                        }
                    ]
                }
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.requests = []  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{server.server_port}"


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


def test_local_patcher_prompt_is_deterministic_and_script_applies_patch(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    base, head = _build_swebench_repo(workspace)
    expected_patch = _git_diff(workspace, base, head, "smoke_module.py")
    _run(workspace, "git", "checkout", "--quiet", base)
    task_file = tmp_path / "task.json"
    task_payload = {
        "case_id": "example__repo-1",
        "problem_statement": "Fix buggy_increment so it adds one.",
        "failing_tests": ["tests/test_smoke_module.py::SmokeModuleTests::test_buggy_increment"],
        "expected_changed_paths": ["smoke_module.py"],
        "memory_context": [
            {
                "title": "buggy_increment in smoke_module.py",
                "summary": "Returns the input without incrementing.",
                "relative_path": "smoke_module.py",
                "symbol": "buggy_increment",
                "score": 1.0,
            }
        ],
    }
    task_file.write_text(json.dumps(task_payload), encoding="utf-8")

    patcher = LocalOpenAIPatcher(
        LocalPatcherConfig(
            base_url="http://unused",
            api_key="test",
            model="qwen-test",
        )
    )
    first_prompt = patcher.build_prompt(task=task_payload, workspace_root=workspace)
    second_prompt = patcher.build_prompt(task=task_payload, workspace_root=workspace)
    assert first_prompt == second_prompt

    server, base_url = _start_fake_openai_server(expected_patch)
    try:
        env = {
            **os.environ,
            "VTM_LOCAL_LLM_BASE_URL": base_url,
            "VTM_LOCAL_LLM_API_KEY": "test",
            "VTM_LOCAL_LLM_MODEL": "qwen-test",
        }
        subprocess.run(
            [
                sys.executable,
                "scripts/vtm_local_patcher.py",
                "--task-file",
                str(task_file),
                "--workspace",
                str(workspace),
            ],
            cwd=Path.cwd(),
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
    finally:
        server.shutdown()

    assert "return value + 1" in (workspace / "smoke_module.py").read_text(encoding="utf-8")
    assert (workspace / ".vtm-local-patcher" / "response.txt").exists()


def test_local_patcher_rejects_invalid_patch_without_mutating_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace-invalid"
    _build_swebench_repo(workspace)
    current = (workspace / "smoke_module.py").read_text(encoding="utf-8")
    patcher = LocalOpenAIPatcher(
        LocalPatcherConfig(
            base_url="http://unused",
            api_key="test",
            model="qwen-test",
        )
    )

    class InvalidPatcher(LocalOpenAIPatcher):
        def _request_patch(self, prompt: str) -> str:
            return "diff --git a/missing.py b/missing.py\n"

    task_file = tmp_path / "task-invalid.json"
    task_file.write_text(
        json.dumps(
            {
                "case_id": "invalid",
                "problem_statement": "Break it.",
                "expected_changed_paths": ["smoke_module.py"],
                "memory_context": [],
            }
        ),
        encoding="utf-8",
    )
    try:
        InvalidPatcher(patcher._config).run(task_file=task_file, workspace=workspace)
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected invalid patch to fail")

    assert (workspace / "smoke_module.py").read_text(encoding="utf-8") == current


def test_scaffold_bridge_builds_bundle_with_memory_and_test_files(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace-bridge"
    _build_swebench_repo(workspace)
    task_payload = {
        "case_id": "example__repo-1",
        "repo_name": "example__repo",
        "commit_pair_id": "example__repo-1",
        "problem_statement": "Fix buggy_increment so it adds one.",
        "failing_tests": ["tests/test_smoke_module.py::SmokeModuleTests::test_buggy_increment"],
        "fail_to_pass_tests": [
            "tests/test_smoke_module.py::SmokeModuleTests::test_buggy_increment"
        ],
        "expected_changed_paths": ["smoke_module.py"],
        "memory_mode": "lexical_rlm_rerank",
        "memory_context": [
            {
                "title": "buggy_increment in smoke_module.py",
                "summary": "Returns the input without incrementing.",
                "relative_path": "smoke_module.py",
                "symbol": "buggy_increment",
                "score": 1.0,
                "status": "verified",
            }
        ],
    }

    bridge = ScaffoldBridge()
    bundle = bridge.build_bundle(task=task_payload, workspace_root=workspace)
    brief = bridge.build_brief(bundle)

    assert bundle["task"]["memory_mode"] == "lexical_rlm_rerank"
    assert bundle["memory_context"][0]["title"] == "buggy_increment in smoke_module.py"
    paths = {item["path"] for item in bundle["relevant_files"]}
    assert "smoke_module.py" in paths
    assert "tests/test_smoke_module.py" in paths
    assert "Memory Context" in brief
    assert "Relevant Files" in brief


def test_scaffold_bridge_cli_writes_bundle_and_runs_delegate(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace-bridge-cli"
    _build_swebench_repo(workspace)
    task_file = tmp_path / "task-bridge.json"
    artifact_root = tmp_path / "artifacts"
    delegate_script = tmp_path / "delegate.py"
    task_file.write_text(
        json.dumps(
            {
                "case_id": "example__repo-1",
                "repo_name": "example__repo",
                "commit_pair_id": "example__repo-1",
                "problem_statement": "Fix buggy_increment so it adds one.",
                "failing_tests": [
                    "tests/test_smoke_module.py::SmokeModuleTests::test_buggy_increment"
                ],
                "expected_changed_paths": ["smoke_module.py"],
                "memory_context": [],
            }
        ),
        encoding="utf-8",
    )
    delegate_script.write_text(
        "from pathlib import Path\n"
        "import argparse\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--bundle', required=True)\n"
        "parser.add_argument('--brief', required=True)\n"
        "args = parser.parse_args()\n"
        "bundle = Path(args.bundle)\n"
        "brief = Path(args.brief)\n"
        "assert bundle.exists()\n"
        "assert brief.exists()\n"
        "print(bundle.name)\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/vtm_scaffold_bridge.py",
            "--task-file",
            str(task_file),
            "--workspace",
            str(workspace),
            "--artifact-root",
            str(artifact_root),
            "--delegate-command",
            (
                f"{sys.executable} {delegate_script} "
                "--bundle {scaffold_bundle} --brief {brief_file}"
            ),
        ],
        cwd=Path.cwd(),
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert (artifact_root / "scaffold-bundle.json").exists()
    assert (artifact_root / "scaffold-brief.md").exists()
    assert (artifact_root / "delegate.stdout").read_text(encoding="utf-8").strip() == (
        "scaffold-bundle.json"
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
) -> None:
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
    executor_command = (
        "python3",
        "-c",
        (
            "from pathlib import Path; "
            "Path('smoke_module.py').write_text("
            "\"def buggy_increment(value: int) -> int:\\n"
            "    return value + 1\\n\", encoding='utf-8')"
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
            executor_command=executor_command,
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
