from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from textwrap import dedent


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
