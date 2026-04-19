from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import vtm.benchmarks.report as report
from vtm.benchmarks import (
    BenchmarkManifest,
    BenchmarkRunConfig,
    BenchmarkRunner,
    BenchmarkRunResult,
    matrix,
    prepare_swebench_lite,
    run,
    subset_manifest,
)


def _cli_env() -> dict[str, str]:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = "src" if not existing else f"src:{existing}"
    return env


def _run_subprocess(*args, **kwargs):  # noqa: ANN002, ANN003
    return subprocess.run(*args, env=_cli_env(), **kwargs)


def test_run_cli_parser_accepts_rlm_execution_args() -> None:
    args = run.build_parser().parse_args(
        [
            "--manifest",
            "benchmarks/manifests/synthetic-smoke.json",
            "--suite",
            "coding",
            "--output",
            "out",
            "--rlm-model-id",
            "gpt-test",
        ]
    )

    assert args.rlm_model_id == "gpt-test"


def test_run_cli_parser_accepts_maintained_coding_engine() -> None:
    args = run.build_parser().parse_args(
        [
            "--manifest",
            "benchmarks/manifests/synthetic-smoke.json",
            "--suite",
            "coding",
            "--output",
            "out",
            "--coding-engine",
            "vendored_rlm",
        ]
    )

    assert args.coding_engine == "vendored_rlm"


def test_run_cli_parser_accepts_new_lexical_modes() -> None:
    args = run.build_parser().parse_args(
        [
            "--manifest",
            "benchmarks/manifests/synthetic-smoke.json",
            "--suite",
            "retrieval",
            "--mode",
            "verified_lexical",
            "--output",
            "out",
        ]
    )

    assert args.mode == "verified_lexical"

    naive = run.build_parser().parse_args(
        [
            "--manifest",
            "benchmarks/manifests/synthetic-smoke.json",
            "--suite",
            "retrieval",
            "--mode",
            "naive_lexical",
            "--output",
            "out",
        ]
    )

    assert naive.mode == "naive_lexical"


def test_run_cli_parser_accepts_seed_on_base_query_on_head() -> None:
    args = run.build_parser().parse_args(
        [
            "--manifest",
            "benchmarks/manifests/synthetic-smoke.json",
            "--suite",
            "retrieval",
            "--output",
            "out",
            "--seed-on-base-query-on-head",
        ]
    )

    assert args.seed_on_base_query_on_head is True


def test_matrix_cli_parser_accepts_rlm_execution_args() -> None:
    args = matrix.build_parser().parse_args(
        [
            "--output",
            "out",
            "--rlm-model-id",
            "gpt-test",
        ]
    )

    assert args.rlm_model_id == "gpt-test"


def test_matrix_cli_parser_accepts_maintained_coding_engine() -> None:
    args = matrix.build_parser().parse_args(
        [
            "--output",
            "out",
            "--coding-engine",
            "vendored_rlm",
        ]
    )

    assert args.coding_engine == "vendored_rlm"


def test_matrix_cli_parser_accepts_seed_on_base_query_on_head() -> None:
    args = matrix.build_parser().parse_args(
        [
            "--output",
            "out",
            "--seed-on-base-query-on-head",
        ]
    )

    assert args.seed_on_base_query_on_head is True


def test_controlled_coding_drift_preset_includes_three_maintained_modes() -> None:
    preset = matrix.PRESETS["controlled_coding_drift"]

    assert preset.suite == "coding"
    assert preset.manifest_path == "benchmarks/manifests/controlled-coding-drift.json"
    assert preset.modes == ("no_memory", "naive_lexical", "verified_lexical")
    assert preset.baseline_mode == "no_memory"


def test_report_cli_requires_at_least_one_run_location(tmp_path: Path) -> None:
    try:
        report.export_paper_tables(
            retrieval_locations=[],
            drift_locations=[],
            coding_locations=[],
            output_dir=tmp_path / "paper-tables",
        )
    except ValueError as exc:
        assert "provide at least one" in str(exc)
    else:
        raise AssertionError("expected paper table export without run locations to fail")


def test_run_cli_help_marks_legacy_docker_backend() -> None:
    help_text = run.build_parser().format_help()

    assert "docker_workspace" in help_text
    assert "legacy/non-" in help_text


def test_prepare_swebench_lite_cli_parser_accepts_skip_failed_instances() -> None:
    args = prepare_swebench_lite.build_parser().parse_args(
        [
            "--output-manifest",
            "out.json",
            "--skip-failed-instances",
        ]
    )

    assert args.skip_failed_instances is True


def test_subset_manifest_cli_parser_accepts_repeated_case_ids() -> None:
    args = subset_manifest.build_parser().parse_args(
        [
            "--input",
            "in.json",
            "--output",
            "out.json",
            "--case-id",
            "astropy__astropy-14365",
            "--case-id",
            "astropy__astropy-14995",
        ]
    )

    assert args.case_id == ["astropy__astropy-14365", "astropy__astropy-14995"]


def test_execute_benchmark_run_resolves_openrouter_defaults(
    monkeypatch,
    tmp_path: Path,
) -> None:
    manifest = BenchmarkManifest.from_path("benchmarks/manifests/synthetic-smoke.json")
    captured: dict[str, object] = {}

    class FakeAdapter:
        def __init__(self, *, model: str, base_url: str, api_key: str | None) -> None:
            captured["adapter_model"] = model
            captured["adapter_base_url"] = base_url
            captured["adapter_api_key"] = api_key

    class FakeRunner:
        def __init__(self, manifest, config, rlm_adapter=None) -> None:  # noqa: ANN001, ANN204
            captured["runner_manifest"] = manifest
            captured["runner_config"] = config
            captured["runner_adapter"] = rlm_adapter

        def run(self) -> BenchmarkRunResult:
            return BenchmarkRunResult(
                run_id="run-1",
                manifest_id="synthetic_python_smoke",
                manifest_digest="deadbeef",
                suite="coding",
                mode="lexical_rlm_rerank",
                case_count=0,
                started_at="2026-01-01T00:00:00Z",
                completed_at="2026-01-01T00:00:01Z",
                metrics={},
                artifacts={},
            )

    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-test-key")
    monkeypatch.setenv("VTM_OPENROUTER_BASE_URL", "https://openrouter.example/api/v1")
    monkeypatch.setenv("VTM_EXECUTION_MODEL", "google/gemma-test")
    monkeypatch.setenv("VTM_RERANK_MODEL", "nvidia/nemotron-test")
    monkeypatch.setattr(run, "OpenAIRLMAdapter", FakeAdapter)
    monkeypatch.setattr(run, "BenchmarkRunner", FakeRunner)

    result = run.execute_benchmark_run(
        manifest,
        BenchmarkRunConfig(
            manifest_path="benchmarks/manifests/synthetic-smoke.json",
            suite="coding",
            mode="lexical_rlm_rerank",
            output_dir=str(tmp_path / "openrouter-defaults"),
        ),
    )

    assert result.run_id == "run-1"
    assert captured["adapter_model"] == "nvidia/nemotron-test"
    assert captured["adapter_base_url"] == "https://openrouter.example/api/v1"
    assert captured["adapter_api_key"] == "openrouter-test-key"
    resolved_config = captured["runner_config"]
    assert isinstance(resolved_config, BenchmarkRunConfig)
    assert resolved_config.rlm_model_id == "google/gemma-test"


def test_benchmark_cli_runs_synthetic_retrieval(tmp_path: Path) -> None:
    output_dir = tmp_path / "cli-run"
    completed = _run_subprocess(
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


def test_benchmark_cli_filters_to_selected_pair(tmp_path: Path) -> None:
    output_dir = tmp_path / "cli-filtered-run"
    completed = _run_subprocess(
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
    completed = _run_subprocess(
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


def test_benchmark_cli_rejects_attempt_flags_for_retrieval(tmp_path: Path) -> None:
    output_dir = tmp_path / "cli-invalid-attempts"
    completed = _run_subprocess(
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
    completed = _run_subprocess(
        [
            sys.executable,
            "-m",
            "vtm.benchmarks.run",
            "--manifest",
            "benchmarks/manifests/synthetic-smoke.json",
            "--suite",
            "coding",
            "--mode",
            "verified_lexical",
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
    completed = _run_subprocess(
        [
            sys.executable,
            "-m",
            "vtm.benchmarks.run",
            "--manifest",
            "benchmarks/manifests/synthetic-smoke.json",
            "--suite",
            "coding",
            "--mode",
            "no_memory",
            "--output",
            str(output_dir),
            "--workspace-backend",
            "docker_workspace",
            "--pair",
            "bugfix",
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


def test_benchmark_compare_cli_reports_retrieval_deltas(tmp_path: Path) -> None:
    baseline_dir = tmp_path / "retrieval-baseline"
    candidate_dir = tmp_path / "retrieval-candidate"
    comparison_dir = tmp_path / "retrieval-comparison"

    _run_subprocess(
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
    _run_subprocess(
        [
            sys.executable,
            "-m",
            "vtm.benchmarks.run",
            "--manifest",
            "benchmarks/manifests/synthetic-smoke.json",
            "--suite",
            "retrieval",
            "--mode",
            "verified_lexical",
            "--output",
            str(candidate_dir),
            "--max-cases",
            "2",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    completed = _run_subprocess(
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


def test_benchmark_compare_cli_reports_coding_attempt_metrics(
    tmp_path: Path,
    install_fake_vendored_rlm,
) -> None:
    def apply_candidate_update(task_pack, workspace_root: Path, artifact_root: Path) -> None:
        attempt_dir = next(
            (
                path
                for path in reversed(artifact_root.parents)
                if path.name.startswith("attempt-")
            ),
            None,
        )
        if attempt_dir is None or attempt_dir.name != "attempt-02":
            return
        diff_result = _run_subprocess(
            [
                "git",
                "diff",
                "--binary",
                "--no-ext-diff",
                f"{task_pack.base_ref}..{task_pack.head_ref}",
            ],
            cwd=workspace_root,
            check=True,
            capture_output=True,
            text=True,
        )
        _run_subprocess(
            ["git", "apply", "--whitespace=nowarn"],
            cwd=workspace_root,
            input=diff_result.stdout,
            check=True,
            capture_output=True,
            text=True,
        )

    install_fake_vendored_rlm(
        apply_workspace_update=lambda task_pack, workspace_root, artifact_root: None
    )
    baseline_dir = tmp_path / "coding-baseline"
    candidate_dir = tmp_path / "coding-candidate"
    comparison_dir = tmp_path / "coding-comparison"
    manifest = BenchmarkManifest.from_path("benchmarks/manifests/synthetic-smoke.json")
    BenchmarkRunner(
        manifest,
        BenchmarkRunConfig(
            manifest_path="benchmarks/manifests/synthetic-smoke.json",
            suite="coding",
            mode="verified_lexical",
            output_dir=str(baseline_dir),
            pair_filters=("bugfix",),
            rlm_model_id="fake-model",
            attempt_count=2,
            pass_k_values=(1, 2),
        ),
    ).run()

    install_fake_vendored_rlm(apply_workspace_update=apply_candidate_update)
    BenchmarkRunner(
        manifest,
        BenchmarkRunConfig(
            manifest_path="benchmarks/manifests/synthetic-smoke.json",
            suite="coding",
            mode="verified_lexical",
            output_dir=str(candidate_dir),
            pair_filters=("bugfix",),
            rlm_model_id="fake-model",
            attempt_count=2,
            pass_k_values=(1, 2),
        ),
    ).run()

    completed = _run_subprocess(
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
    completed = _run_subprocess(
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
            "verified_lexical",
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
    assert set(payload["run_results"]) == {"no_memory", "verified_lexical"}
    assert set(payload["comparison_results"]) == {"verified_lexical"}
    assert (output_dir / "matrix.json").exists()
    assert (output_dir / "runs" / "no_memory" / "summary.json").exists()
    assert (
        output_dir / "comparisons" / "no_memory-vs-verified_lexical" / "comparison.json"
    ).exists()


def test_benchmark_matrix_preset_runs_maintained_retrieval_modes(tmp_path: Path) -> None:
    output_dir = tmp_path / "retrieval-preset"
    args = matrix.build_parser().parse_args(
        [
            "--preset",
            "synthetic_retrieval",
            "--output",
            str(output_dir),
            "--max-cases",
            "1",
            "--comparison-bootstrap-samples",
            "100",
        ]
    )

    result = matrix.run_matrix_from_args(args)

    assert result.suite == "retrieval"
    assert result.baseline_mode == "no_memory"
    assert result.modes == ("no_memory", "naive_lexical", "verified_lexical")
    assert set(result.comparison_results) == {"naive_lexical", "verified_lexical"}
    assert (output_dir / "runs" / "naive_lexical" / "summary.json").exists()


def test_benchmark_matrix_preset_runs_maintained_drifted_retrieval_modes(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "retrieval-drifted-preset"
    args = matrix.build_parser().parse_args(
        [
            "--preset",
            "synthetic_retrieval_drifted",
            "--output",
            str(output_dir),
            "--max-cases",
            "1",
            "--comparison-bootstrap-samples",
            "100",
        ]
    )

    result = matrix.run_matrix_from_args(args)
    manifest_lock = json.loads(
        (output_dir / "runs" / "verified_lexical" / "manifest.lock.json").read_text(
            encoding="utf-8"
        )
    )

    assert result.suite == "retrieval"
    assert result.baseline_mode == "no_memory"
    assert result.modes == ("no_memory", "naive_lexical", "verified_lexical")
    assert set(result.comparison_results) == {"naive_lexical", "verified_lexical"}
    assert manifest_lock["seed_on_base_query_on_head"] is True


def test_benchmark_matrix_preset_runs_verified_drift_mode_only(tmp_path: Path) -> None:
    output_dir = tmp_path / "drift-preset"
    args = matrix.build_parser().parse_args(
        [
            "--preset",
            "synthetic_drift",
            "--output",
            str(output_dir),
            "--max-cases",
            "1",
        ]
    )

    result = matrix.run_matrix_from_args(args)

    assert result.suite == "drift"
    assert result.baseline_mode == "verified_lexical"
    assert result.modes == ("verified_lexical",)
    assert result.comparison_results == {}
    assert (output_dir / "runs" / "verified_lexical" / "summary.json").exists()


def test_benchmark_matrix_preset_runs_maintained_coding_modes(
    tmp_path: Path,
    install_fake_vendored_rlm,
) -> None:
    install_fake_vendored_rlm()
    output_dir = tmp_path / "coding-matrix"
    args = matrix.build_parser().parse_args(
        [
            "--preset",
            "synthetic_coding",
            "--output",
            str(output_dir),
            "--pair",
            "bugfix",
            "--max-cases",
            "1",
            "--rlm-model-id",
            "fake-model",
            "--comparison-bootstrap-samples",
            "100",
        ]
    )

    result = matrix.run_matrix_from_args(args)

    assert result.suite == "coding"
    assert result.modes == ("no_memory", "naive_lexical", "verified_lexical")
    assert set(result.comparison_results) == {"naive_lexical", "verified_lexical"}
    assert (output_dir / "runs" / "naive_lexical" / "summary.json").exists()
