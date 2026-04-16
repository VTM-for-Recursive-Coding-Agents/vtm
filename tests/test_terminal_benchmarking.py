from __future__ import annotations

import json
import subprocess
from pathlib import Path

from vtm.benchmarks import BenchmarkManifest, BenchmarkRunConfig, BenchmarkRunner
from vtm.benchmarks.synthetic_terminal import SyntheticTerminalSmokeCorpus
from vtm.harness.models import HarnessTaskPack
from vtm.services import TransactionalMemoryKernel


def _run(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        list(args),
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def test_terminal_smoke_manifest_tasks_fail_on_base_and_pass_on_head(tmp_path: Path) -> None:
    manifest = BenchmarkManifest.from_path("benchmarks/manifests/terminal-smoke.json")
    repo_root = tmp_path / "terminal-corpus"
    SyntheticTerminalSmokeCorpus().materialize(repo_root)
    pair_map = {
        pair.pair_id: pair
        for repo in manifest.repos
        if repo.repo_name == "synthetic_terminal_smoke"
        for pair in repo.commit_pairs
    }
    difficulty_counts: dict[str, int] = {}
    for task in manifest.coding_tasks:
        difficulty = task.difficulty or "unknown"
        difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1
        pair = pair_map[task.commit_pair_id]
        _run(repo_root, "git", "checkout", "--quiet", pair.base_ref)
        base_result = subprocess.run(
            list(task.test_command),
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
        _run(repo_root, "git", "checkout", "--quiet", pair.head_ref)
        head_result = subprocess.run(
            list(task.test_command),
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
        assert base_result.returncode != 0, task.case_id
        assert head_result.returncode == 0, task.case_id

    assert len(manifest.coding_tasks) == 15
    assert difficulty_counts == {"easy": 5, "medium": 5, "hard": 5}


def test_terminal_shell_smoke_manifest_tasks_fail_on_base_and_pass_on_head(
    tmp_path: Path,
) -> None:
    manifest = BenchmarkManifest.from_path("benchmarks/manifests/terminal-shell-smoke.json")
    repo_root = tmp_path / "terminal-shell-corpus"
    SyntheticTerminalSmokeCorpus().materialize(repo_root)
    pair_map = {
        pair.pair_id: pair
        for repo in manifest.repos
        if repo.repo_name == "synthetic_terminal_smoke"
        for pair in repo.commit_pairs
    }
    difficulty_counts: dict[str, int] = {}
    for task in manifest.coding_tasks:
        difficulty = task.difficulty or "unknown"
        difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1
        assert task.execution_style == "shell_command"
        pair = pair_map[task.commit_pair_id]
        _run(repo_root, "git", "checkout", "--quiet", pair.base_ref)
        base_result = subprocess.run(
            list(task.test_command),
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
        _run(repo_root, "git", "checkout", "--quiet", pair.head_ref)
        head_result = subprocess.run(
            list(task.test_command),
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
        assert base_result.returncode != 0, task.case_id
        assert head_result.returncode == 0, task.case_id

    assert len(manifest.coding_tasks) == 12
    assert difficulty_counts == {"easy": 4, "medium": 4, "hard": 4}


def test_attempt_rows_and_pass_at_k_aggregate_external_executor(tmp_path: Path) -> None:
    manifest = BenchmarkManifest.from_path("benchmarks/manifests/synthetic-smoke.json")
    executor_command = (
        "python3",
        "-c",
        (
            "from pathlib import Path; import sys; "
            "attempt = int(sys.argv[1]); "
            "artifact_root = Path(sys.argv[2]); "
            "artifact_root.mkdir(parents=True, exist_ok=True); "
            "(artifact_root / 'attempt.txt').write_text(str(attempt), encoding='utf-8'); "
            "Path('bugfix_module.py').write_text("
            "\"def buggy_increment(value: int) -> int:\\n"
            "    \\\"\\\"\\\"Return value plus one.\\\"\\\"\\\"\\n"
            "    return value + 1\\n\", "
            "encoding='utf-8') if attempt == 2 else None"
        ),
        "{attempt}",
        "{artifact_root}",
    )
    result = BenchmarkRunner(
        manifest,
        BenchmarkRunConfig(
            manifest_path="benchmarks/manifests/synthetic-smoke.json",
            suite="coding",
            mode="lexical",
            output_dir=str(tmp_path / "attempted-coding"),
            pair_filters=("bugfix",),
            max_cases=1,
            executor_command=executor_command,
            attempt_count=3,
            pass_k_values=(1, 2, 3),
        ),
    ).run()

    attempt_lines = Path(result.artifacts["attempts_jsonl"]).read_text(
        encoding="utf-8"
    ).splitlines()
    attempt_rows = [json.loads(line) for line in attempt_lines]
    result_row = json.loads(
        Path(result.artifacts["results_jsonl"]).read_text(encoding="utf-8").splitlines()[0]
    )

    assert len(attempt_rows) == 3
    assert result_row["metadata"]["attempt_count"] == 3
    assert result_row["metadata"]["successful_attempt_indices"] == [2]
    assert result_row["metadata"]["best_attempt_index"] == 2
    assert result.metrics["pass_at_1"] == 0.0
    assert result.metrics["pass_at_2"] == 1.0
    assert result.metrics["pass_at_3"] == 1.0
    assert result.metrics["resolved_at_2"] == 1.0
    assert result.metrics["patch_applied_at_2"] == 1.0
    assert result.metrics["mean_attempts_executed"] == 3.0
    assert attempt_rows[1]["metadata"]["artifact_root"].endswith("attempt-02")
    assert (
        Path(attempt_rows[1]["metadata"]["artifact_root"]) / "attempt.txt"
    ).read_text(encoding="utf-8") == "2"


def test_retrieval_query_overrides_default_query(tmp_path: Path, monkeypatch) -> None:
    captured_queries: list[str] = []
    original_retrieve = TransactionalMemoryKernel.retrieve

    def wrapped_retrieve(self, request):
        captured_queries.append(request.query)
        return original_retrieve(self, request)

    monkeypatch.setattr(TransactionalMemoryKernel, "retrieve", wrapped_retrieve)
    manifest = BenchmarkManifest.from_path("benchmarks/manifests/terminal-smoke.json")
    result = BenchmarkRunner(
        manifest,
        BenchmarkRunConfig(
            manifest_path="benchmarks/manifests/terminal-smoke.json",
            suite="coding",
            mode="lexical",
            output_dir=str(tmp_path / "retrieval-query"),
            pair_filters=("export_path",),
            max_cases=1,
        ),
    ).run()

    task_pack = HarnessTaskPack.from_json(
        (
            tmp_path
            / "retrieval-query"
            / "task-packs"
            / "terminal_export_path_unittest.json"
        ).read_text(encoding="utf-8")
    )

    assert result.case_count == 1
    assert captured_queries[0] == "build export path lowercase json report name"
    assert task_pack.retrieval_query == captured_queries[0]


def test_shell_command_attempts_run_under_docker_workspace_external_executor(
    tmp_path: Path,
    fake_docker_binary: Path,
) -> None:
    manifest = BenchmarkManifest.from_path("benchmarks/manifests/terminal-shell-smoke.json")
    result = BenchmarkRunner(
        manifest,
        BenchmarkRunConfig(
            manifest_path="benchmarks/manifests/terminal-shell-smoke.json",
            suite="coding",
            mode="no_memory",
            output_dir=str(tmp_path / "docker-shell-external"),
            pair_filters=("shell_daily_report",),
            max_cases=1,
            workspace_backend="docker_workspace",
            docker_image="python:3.12",
            docker_binary=str(fake_docker_binary),
            executor_command=("python3", "scripts/build_daily_report.py"),
            attempt_count=2,
            pass_k_values=(1, 2),
        ),
    ).run()

    attempt_rows = [
        json.loads(line)
        for line in Path(result.artifacts["attempts_jsonl"]).read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    result_row = json.loads(
        Path(result.artifacts["results_jsonl"]).read_text(encoding="utf-8").splitlines()[0]
    )
    manifest_lock = json.loads(
        Path(result.artifacts["manifest_lock"]).read_text(encoding="utf-8")
    )

    assert result.case_count == 1
    assert len(attempt_rows) == 2
    assert all(
        row["metadata"]["workspace_backend"] == "docker_workspace" for row in attempt_rows
    )
    assert all(row["metadata"]["docker_network"] == "none" for row in attempt_rows)
    assert result_row["metadata"]["execution_style"] == "shell_command"
    assert result.metrics["pass_at_1"] == 1.0
    assert result.metrics["workspace_backend_breakdown"] == {"docker_workspace": 1}
    assert result.metrics["execution_style_metrics"]["shell_command"]["pass_at_1"] == 1.0
    assert manifest_lock["workspace_backend"] == "docker_workspace"
    assert manifest_lock["docker_image"] == "python:3.12"
