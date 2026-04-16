from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

from vtm.benchmarks import matrix, run


def test_run_cli_parser_accepts_rlm_coding_executor() -> None:
    args = run.build_parser().parse_args(
        [
            "--manifest",
            "benchmarks/manifests/synthetic-smoke.json",
            "--suite",
            "coding",
            "--output",
            "out",
            "--coding-executor",
            "rlm",
        ]
    )

    assert args.coding_executor == "rlm"


def test_matrix_cli_parser_accepts_rlm_coding_executor() -> None:
    args = matrix.build_parser().parse_args(
        [
            "--output",
            "out",
            "--coding-executor",
            "rlm",
        ]
    )

    assert args.coding_executor == "rlm"


def test_benchmark_cli_runs_synthetic_retrieval(tmp_path: Path) -> None:
    output_dir = tmp_path / "cli-run"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "vtm.benchmarks.run",
            "--manifest",
            "benchmarks/manifests/synthetic-smoke.json",
            "--suite",
            "retrieval",
            "--mode",
            "no_memory",
            "--output",
            str(output_dir),
            "--max-cases",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["suite"] == "retrieval"
    assert payload["case_count"] == 1
    assert (output_dir / "summary.json").exists()


def test_benchmark_cli_runs_synthetic_embedding_retrieval(tmp_path: Path) -> None:
    output_dir = tmp_path / "cli-embedding-run"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "vtm.benchmarks.run",
            "--manifest",
            "benchmarks/manifests/synthetic-smoke.json",
            "--suite",
            "retrieval",
            "--mode",
            "embedding",
            "--output",
            str(output_dir),
            "--max-cases",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["mode"] == "embedding"
    assert payload["case_count"] == 1
    assert (output_dir / "summary.json").exists()


def test_benchmark_cli_filters_to_selected_pair(tmp_path: Path) -> None:
    output_dir = tmp_path / "cli-filtered-run"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "vtm.benchmarks.run",
            "--manifest",
            "benchmarks/manifests/synthetic-smoke.json",
            "--suite",
            "retrieval",
            "--mode",
            "no_memory",
            "--output",
            str(output_dir),
            "--pair",
            "stable",
            "--max-cases",
            "2",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    cases = [
        json.loads(line)
        for line in (output_dir / "cases.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert payload["case_count"] == 2
    assert {case["commit_pair_id"] for case in cases} == {"stable"}


def test_benchmark_cli_rejects_unknown_repo_filter(tmp_path: Path) -> None:
    output_dir = tmp_path / "cli-bad-filter"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "vtm.benchmarks.run",
            "--manifest",
            "benchmarks/manifests/synthetic-smoke.json",
            "--suite",
            "retrieval",
            "--mode",
            "no_memory",
            "--output",
            str(output_dir),
            "--repo",
            "missing_repo",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "unknown benchmark repos" in completed.stderr


def test_benchmark_cli_runs_attempt_aware_coding_suite(tmp_path: Path) -> None:
    patcher_script = tmp_path / "attempt_patcher.py"
    patcher_script.write_text(
        dedent(
            '''
            from pathlib import Path
            import argparse

            parser = argparse.ArgumentParser()
            parser.add_argument('--attempt', type=int, required=True)
            parser.add_argument('--artifact-root', required=True)
            args = parser.parse_args()
            artifact_root = Path(args.artifact_root)
            artifact_root.mkdir(parents=True, exist_ok=True)
            (artifact_root / 'attempt.txt').write_text(str(args.attempt), encoding='utf-8')
            if args.attempt == 2:
                Path('bugfix_module.py').write_text(
                    'def buggy_increment(value: int) -> int:\\n'
                    '    """Return value plus one."""\\n'
                    '    return value + 1\\n',
                    encoding='utf-8',
                )
            '''
        ).lstrip(),
        encoding="utf-8",
    )
    output_dir = tmp_path / "cli-coding-attempts"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "vtm.benchmarks.run",
            "--manifest",
            "benchmarks/manifests/synthetic-smoke.json",
            "--suite",
            "coding",
            "--mode",
            "lexical",
            "--output",
            str(output_dir),
            "--pair",
            "bugfix",
            "--attempts",
            "2",
            "--pass-k",
            "1",
            "--pass-k",
            "2",
            "--executor-command",
            (
                f"python3 {patcher_script} "
                "--attempt {attempt} --artifact-root {artifact_root}"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["suite"] == "coding"
    assert payload["metrics"]["pass_at_1"] == 0.0
    assert payload["metrics"]["pass_at_2"] == 1.0
    assert (output_dir / "attempts.jsonl").exists()


def test_benchmark_cli_rejects_attempt_flags_for_retrieval(tmp_path: Path) -> None:
    output_dir = tmp_path / "cli-invalid-attempts"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "vtm.benchmarks.run",
            "--manifest",
            "benchmarks/manifests/synthetic-smoke.json",
            "--suite",
            "retrieval",
            "--mode",
            "no_memory",
            "--output",
            str(output_dir),
            "--attempts",
            "2",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "--attempts > 1 is only supported for coding suites" in completed.stderr


def test_benchmark_cli_rejects_local_backend_docker_flags(tmp_path: Path) -> None:
    output_dir = tmp_path / "cli-invalid-docker-local"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "vtm.benchmarks.run",
            "--manifest",
            "benchmarks/manifests/synthetic-smoke.json",
            "--suite",
            "coding",
            "--mode",
            "lexical",
            "--output",
            str(output_dir),
            "--docker-image",
            "python:3.12",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert (
        "--docker-image is only supported with --workspace-backend docker_workspace"
        in completed.stderr
    )


def test_benchmark_cli_rejects_docker_backend_without_image(tmp_path: Path) -> None:
    output_dir = tmp_path / "cli-invalid-docker-image"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "vtm.benchmarks.run",
            "--manifest",
            "benchmarks/manifests/terminal-shell-smoke.json",
            "--suite",
            "coding",
            "--mode",
            "no_memory",
            "--output",
            str(output_dir),
            "--workspace-backend",
            "docker_workspace",
            "--pair",
            "shell_daily_report",
            "--executor-command",
            "python3 scripts/build_daily_report.py",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert (
        "--docker-image is required when --workspace-backend docker_workspace"
        in completed.stderr
    )


def test_benchmark_cli_runs_shell_command_track_with_docker_backend(
    tmp_path: Path,
    fake_docker_binary: Path,
) -> None:
    output_dir = tmp_path / "cli-docker-shell"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "vtm.benchmarks.run",
            "--manifest",
            "benchmarks/manifests/terminal-shell-smoke.json",
            "--suite",
            "coding",
            "--mode",
            "no_memory",
            "--output",
            str(output_dir),
            "--workspace-backend",
            "docker_workspace",
            "--docker-image",
            "python:3.12",
            "--docker-binary",
            str(fake_docker_binary),
            "--pair",
            "shell_daily_report",
            "--max-cases",
            "1",
            "--executor-command",
            "python3 scripts/build_daily_report.py",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["suite"] == "coding"
    assert payload["metrics"]["pass_at_1"] == 1.0
    assert payload["metrics"]["workspace_backend_breakdown"] == {"docker_workspace": 1}
    assert (output_dir / "attempts.jsonl").exists()


def test_benchmark_compare_cli_reports_retrieval_deltas(tmp_path: Path) -> None:
    baseline_dir = tmp_path / "retrieval-baseline"
    candidate_dir = tmp_path / "retrieval-candidate"
    comparison_dir = tmp_path / "retrieval-comparison"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "vtm.benchmarks.run",
            "--manifest",
            "benchmarks/manifests/synthetic-smoke.json",
            "--suite",
            "retrieval",
            "--mode",
            "no_memory",
            "--output",
            str(baseline_dir),
            "--max-cases",
            "2",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "vtm.benchmarks.run",
            "--manifest",
            "benchmarks/manifests/synthetic-smoke.json",
            "--suite",
            "retrieval",
            "--mode",
            "lexical",
            "--output",
            str(candidate_dir),
            "--max-cases",
            "2",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "vtm.benchmarks.compare",
            "--baseline",
            str(baseline_dir),
            "--candidate",
            str(candidate_dir),
            "--output",
            str(comparison_dir),
            "--bootstrap-samples",
            "100",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["suite"] == "retrieval"
    assert payload["common_case_count"] == 2
    assert payload["metrics"]["summary_scalar_deltas"]["recall_at_5"]["delta"] >= 0.0
    assert payload["metrics"]["paired_numeric_metrics"]["recall_at_5"]["candidate_mean"] >= 0.0
    assert (comparison_dir / "comparison.json").exists()
    assert (comparison_dir / "comparison.md").exists()


def test_benchmark_compare_cli_reports_coding_attempt_metrics(tmp_path: Path) -> None:
    failing_script = tmp_path / "failing_patcher.py"
    failing_script.write_text(
        dedent(
            '''
            import argparse
            from pathlib import Path

            parser = argparse.ArgumentParser()
            parser.add_argument('--attempt', type=int, required=True)
            parser.add_argument('--artifact-root', required=True)
            args = parser.parse_args()
            artifact_root = Path(args.artifact_root)
            artifact_root.mkdir(parents=True, exist_ok=True)
            (artifact_root / 'attempt.txt').write_text(str(args.attempt), encoding='utf-8')
            '''
        ).lstrip(),
        encoding="utf-8",
    )
    succeeding_script = tmp_path / "succeeding_patcher.py"
    succeeding_script.write_text(
        dedent(
            '''
            import argparse
            from pathlib import Path

            parser = argparse.ArgumentParser()
            parser.add_argument('--attempt', type=int, required=True)
            parser.add_argument('--artifact-root', required=True)
            args = parser.parse_args()
            artifact_root = Path(args.artifact_root)
            artifact_root.mkdir(parents=True, exist_ok=True)
            (artifact_root / 'attempt.txt').write_text(str(args.attempt), encoding='utf-8')
            if args.attempt == 2:
                Path('bugfix_module.py').write_text(
                    'def buggy_increment(value: int) -> int:\\n'
                    '    """Return value plus one."""\\n'
                    '    return value + 1\\n',
                    encoding='utf-8',
                )
            '''
        ).lstrip(),
        encoding="utf-8",
    )

    baseline_dir = tmp_path / "coding-baseline"
    candidate_dir = tmp_path / "coding-candidate"
    comparison_dir = tmp_path / "coding-comparison"
    for output_dir, script_path in (
        (baseline_dir, failing_script),
        (candidate_dir, succeeding_script),
    ):
        subprocess.run(
            [
                sys.executable,
                "-m",
                "vtm.benchmarks.run",
                "--manifest",
                "benchmarks/manifests/synthetic-smoke.json",
                "--suite",
                "coding",
                "--mode",
                "lexical",
                "--output",
                str(output_dir),
                "--pair",
                "bugfix",
                "--attempts",
                "2",
                "--pass-k",
                "1",
                "--pass-k",
                "2",
                "--executor-command",
                (
                    f"python3 {script_path} "
                    "--attempt {attempt} --artifact-root {artifact_root}"
                ),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "vtm.benchmarks.compare",
            "--baseline",
            str(baseline_dir),
            "--candidate",
            str(candidate_dir),
            "--output",
            str(comparison_dir),
            "--bootstrap-samples",
            "100",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    pass_at_2 = payload["metrics"]["paired_attempt_binary_metrics"]["pass_at_2"]
    assert payload["suite"] == "coding"
    assert payload["common_case_count"] == 1
    assert pass_at_2["baseline_rate"] == 0.0
    assert pass_at_2["candidate_rate"] == 1.0
    assert pass_at_2["candidate_only_true_count"] == 1
    assert (comparison_dir / "comparison.json").exists()
    assert (comparison_dir / "comparison.md").exists()


def test_benchmark_matrix_cli_runs_manual_retrieval_matrix(tmp_path: Path) -> None:
    output_dir = tmp_path / "retrieval-matrix"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "vtm.benchmarks.matrix",
            "--manifest",
            "benchmarks/manifests/synthetic-smoke.json",
            "--suite",
            "retrieval",
            "--output",
            str(output_dir),
            "--mode",
            "no_memory",
            "--mode",
            "lexical",
            "--max-cases",
            "1",
            "--comparison-bootstrap-samples",
            "100",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["suite"] == "retrieval"
    assert payload["baseline_mode"] == "no_memory"
    assert set(payload["run_results"]) == {"no_memory", "lexical"}
    assert set(payload["comparison_results"]) == {"lexical"}
    assert (output_dir / "matrix.json").exists()
    assert (output_dir / "runs" / "no_memory" / "summary.json").exists()
    assert (output_dir / "comparisons" / "no_memory-vs-lexical" / "comparison.json").exists()


def test_benchmark_matrix_cli_runs_terminal_smoke_preset(tmp_path: Path) -> None:
    output_dir = tmp_path / "terminal-smoke-matrix"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "vtm.benchmarks.matrix",
            "--preset",
            "terminal_smoke",
            "--output",
            str(output_dir),
            "--mode",
            "no_memory",
            "--mode",
            "lexical",
            "--max-cases",
            "1",
            "--comparison-bootstrap-samples",
            "100",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["preset_name"] == "terminal_smoke"
    assert payload["manifest_path"] == "benchmarks/manifests/terminal-smoke.json"
    assert payload["suite"] == "coding"
    assert (output_dir / "matrix.md").exists()
    assert (output_dir / "comparisons" / "no_memory-vs-lexical" / "comparison.md").exists()
