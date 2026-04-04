from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


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
