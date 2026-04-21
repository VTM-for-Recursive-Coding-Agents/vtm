from __future__ import annotations

import csv
import json
from pathlib import Path

from vtm.benchmarks import BenchmarkManifest, BenchmarkRunConfig, BenchmarkRunner
from vtm.benchmarks import report as benchmark_report
from vtm.benchmarks.models import BenchmarkAttemptResult, BenchmarkCaseResult, BenchmarkRunResult
from vtm.benchmarks.reporting import BenchmarkReporter
from vtm.enums import ValidityStatus


def _write_summary(
    path: Path,
    *,
    suite: str,
    mode: str = "verified_lexical",
    case_count: int = 1,
    manifest_id: str = "synthetic_python_smoke",
    metrics: dict[str, object] | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        BenchmarkRunResult(
            run_id=f"{path.parent.name}-run",
            manifest_id=manifest_id,
            manifest_digest="deadbeef",
            suite=suite,
            mode=mode,
            case_count=case_count,
            started_at="2026-01-01T00:00:00Z",
            completed_at="2026-01-01T00:00:01Z",
            metrics=metrics or {},
            artifacts={"summary_json": str(path)},
        ).to_json(),
        encoding="utf-8",
    )
    return path


def test_benchmark_reporter_computes_safe_retrieval_metrics() -> None:
    reporter = BenchmarkReporter()
    results = [
        BenchmarkCaseResult(
            suite="retrieval",
            mode="verified_lexical",
            case_id="verified-hit",
            repo_name="synthetic_python_smoke",
            commit_pair_id="stable",
            metrics={
                "recall_at_1": 1.0,
                "recall_at_3": 1.0,
                "recall_at_5": 1.0,
                "mrr": 1.0,
                "ndcg": 1.0,
                "latency_ms": 10.0,
                "artifact_bytes_per_memory": 100.0,
                "rank": 1,
            },
            metadata={"expected_head_status": ValidityStatus.VERIFIED.value},
        ),
        BenchmarkCaseResult(
            suite="retrieval",
            mode="verified_lexical",
            case_id="relocated-hit-at-5",
            repo_name="synthetic_python_smoke",
            commit_pair_id="relocated",
            metrics={
                "recall_at_1": 0.0,
                "recall_at_3": 1.0,
                "recall_at_5": 1.0,
                "mrr": 1.0 / 3.0,
                "ndcg": 1.0 / 2.0,
                "latency_ms": 12.0,
                "artifact_bytes_per_memory": 100.0,
                "rank": 3,
            },
            metadata={"expected_head_status": ValidityStatus.RELOCATED.value},
        ),
        BenchmarkCaseResult(
            suite="retrieval",
            mode="verified_lexical",
            case_id="stale-rejected",
            repo_name="synthetic_python_smoke",
            commit_pair_id="deleted",
            metrics={
                "recall_at_1": 0.0,
                "recall_at_3": 0.0,
                "recall_at_5": 0.0,
                "mrr": 0.0,
                "ndcg": 0.0,
                "latency_ms": 11.0,
                "artifact_bytes_per_memory": 100.0,
                "rank": None,
            },
            metadata={"expected_head_status": ValidityStatus.STALE.value},
        ),
        BenchmarkCaseResult(
            suite="retrieval",
            mode="verified_lexical",
            case_id="stale-hit",
            repo_name="synthetic_python_smoke",
            commit_pair_id="deleted",
            metrics={
                "recall_at_1": 0.0,
                "recall_at_3": 1.0,
                "recall_at_5": 1.0,
                "mrr": 0.5,
                "ndcg": 1.0 / 1.584962500721156,
                "latency_ms": 14.0,
                "artifact_bytes_per_memory": 100.0,
                "rank": 2,
            },
            metadata={"expected_head_status": ValidityStatus.STALE.value},
        ),
    ]

    summary = reporter.summarize_results("retrieval", results)

    assert summary["valid_recall_at_1"] == 0.5
    assert summary["valid_recall_at_5"] == 1.0
    assert summary["stale_rejection_rate"] == 0.5
    assert summary["stale_hit_rate"] == 0.5
    assert summary["safe_retrieval_at_1"] == 0.5
    assert summary["safe_retrieval_at_5"] == 0.75


def test_export_paper_tables_writes_suite_csvs_and_combined_markdown(
    tmp_path: Path,
    install_fake_benchmark_agent,
) -> None:
    install_fake_benchmark_agent()
    manifest = BenchmarkManifest.from_path("benchmarks/manifests/synthetic-smoke.json")

    retrieval_run = BenchmarkRunner(
        manifest,
        BenchmarkRunConfig(
            manifest_path="benchmarks/manifests/synthetic-smoke.json",
            suite="retrieval",
            mode="verified_lexical",
            output_dir=str(tmp_path / "retrieval"),
            max_cases=1,
        ),
    ).run()
    drift_run = BenchmarkRunner(
        manifest,
        BenchmarkRunConfig(
            manifest_path="benchmarks/manifests/synthetic-smoke.json",
            suite="drift",
            mode="verified_lexical",
            output_dir=str(tmp_path / "drift"),
            max_cases=1,
        ),
    ).run()
    coding_run = BenchmarkRunner(
        manifest,
        BenchmarkRunConfig(
            manifest_path="benchmarks/manifests/synthetic-smoke.json",
            suite="coding",
            mode="verified_lexical",
            output_dir=str(tmp_path / "coding"),
            pair_filters=("bugfix",),
            max_cases=1,
            execution_model_id="fake-model",
        ),
    ).run()

    artifacts = benchmark_report.export_paper_tables(
        retrieval_locations=[str(Path(retrieval_run.artifacts["summary_json"]))],
        drift_locations=[str(Path(drift_run.artifacts["summary_json"]))],
        coding_locations=[str(Path(coding_run.artifacts["summary_json"]))],
        output_dir=tmp_path / "paper-tables",
    )

    retrieval_rows = list(
        csv.DictReader(Path(artifacts["retrieval_csv"]).read_text(encoding="utf-8").splitlines())
    )
    drift_rows = list(
        csv.DictReader(Path(artifacts["drift_csv"]).read_text(encoding="utf-8").splitlines())
    )
    coding_rows = list(
        csv.DictReader(Path(artifacts["coding_csv"]).read_text(encoding="utf-8").splitlines())
    )
    markdown = Path(artifacts["markdown"]).read_text(encoding="utf-8")
    metadata = json.loads(Path(artifacts["metadata_json"]).read_text(encoding="utf-8"))

    assert retrieval_rows[0]["suite"] == "retrieval"
    assert retrieval_rows[0]["corpus"] == "Synthetic"
    assert retrieval_rows[0]["run_label"] == "retrieval"
    assert retrieval_rows[0]["method"] == "Verified Lexical"
    assert "recall_at_1" in retrieval_rows[0]
    assert "median_latency_ms" in retrieval_rows[0]
    assert drift_rows[0]["suite"] == "drift"
    assert drift_rows[0]["corpus"] == "Synthetic"
    assert drift_rows[0]["run_label"] == "drift"
    assert drift_rows[0]["status_confusion"]
    assert coding_rows[0]["suite"] == "coding"
    assert coding_rows[0]["corpus"] == "Synthetic"
    assert coding_rows[0]["run_label"] == "coding"
    assert coding_rows[0]["pass_at_1"] == "1.0"
    assert "## Retrieval" in markdown
    assert "## Drift" in markdown
    assert "## Coding" in markdown
    assert "| Corpus | Method | Cases |" in markdown
    assert "| Synthetic | Verified Lexical | 1 |" in markdown
    assert metadata["generated_at"]
    assert metadata["input_summary_json_paths"]["retrieval"] == [
        str(Path(retrieval_run.artifacts["summary_json"]))
    ]
    assert metadata["input_summary_json_paths"]["drift"] == [
        str(Path(drift_run.artifacts["summary_json"]))
    ]
    assert metadata["input_summary_json_paths"]["coding"] == [
        str(Path(coding_run.artifacts["summary_json"]))
    ]
    assert "git_branch" in metadata
    assert "git_commit_sha" in metadata


def test_export_paper_tables_supports_controlled_coding_drift_runs(
    tmp_path: Path,
    install_fake_benchmark_agent,
) -> None:
    install_fake_benchmark_agent()
    manifest = BenchmarkManifest.from_path("benchmarks/manifests/controlled-coding-drift.json")

    run_paths: list[str] = []
    for mode in ("no_memory", "naive_lexical", "verified_lexical"):
        run_result = BenchmarkRunner(
            manifest,
            BenchmarkRunConfig(
                manifest_path="benchmarks/manifests/controlled-coding-drift.json",
                suite="coding",
                mode=mode,
                output_dir=str(tmp_path / "controlled-coding-drift-super" / "runs" / mode),
                pair_filters=("stale_api_name",),
                max_cases=1,
                top_k=5,
                execution_model_id="fake-model",
            ),
        ).run()
        run_paths.append(str(Path(run_result.artifacts["summary_json"])))

    artifacts = benchmark_report.export_paper_tables(
        retrieval_locations=[],
        drift_locations=[],
        coding_locations=run_paths,
        output_dir=tmp_path / "paper-tables-controlled",
    )

    coding_rows = list(
        csv.DictReader(Path(artifacts["coding_csv"]).read_text(encoding="utf-8").splitlines())
    )
    metadata = json.loads(Path(artifacts["metadata_json"]).read_text(encoding="utf-8"))

    assert [row["mode"] for row in coding_rows] == [
        "no_memory",
        "naive_lexical",
        "verified_lexical",
    ]
    assert [row["method"] for row in coding_rows] == [
        "No Memory",
        "Naive Lexical",
        "Verified Lexical",
    ]
    assert all(row["corpus"] == "Controlled Coding Drift" for row in coding_rows)
    assert metadata["input_summary_json_paths"]["coding"] == run_paths


def test_export_paper_tables_derives_run_label_and_corpus_from_matrix_layout(
    tmp_path: Path,
) -> None:
    summary_path = _write_summary(
        tmp_path / "matrix-drift" / "runs" / "verified_lexical" / "summary.json",
        suite="drift",
        metrics={
            "relocation_precision": 1.0,
            "relocation_recall": 1.0,
            "false_verified_rate": 0.0,
            "median_verification_latency_ms": 12.0,
            "status_confusion": {"relocated->relocated": 1},
        },
    )

    artifacts = benchmark_report.export_paper_tables(
        retrieval_locations=[],
        drift_locations=[str(summary_path)],
        coding_locations=[],
        output_dir=tmp_path / "paper-tables",
    )

    drift_rows = list(
        csv.DictReader(Path(artifacts["drift_csv"]).read_text(encoding="utf-8").splitlines())
    )
    markdown = Path(artifacts["markdown"]).read_text(encoding="utf-8")

    assert drift_rows[0]["run_label"] == "matrix-drift"
    assert drift_rows[0]["corpus"] == "Synthetic"
    assert (
        "| Corpus | Cases | Relocation Precision | Relocation Recall | "
        "False Verified Rate | Median Verification Latency (ms) |"
    ) in markdown
    assert "| Synthetic | 1 | 1.000 | 1.000 | 0.000 | 12.0 |" in markdown


def test_export_paper_tables_includes_drifted_retrieval_metrics(
    tmp_path: Path,
) -> None:
    manifest = BenchmarkManifest.from_path("benchmarks/manifests/synthetic-smoke.json")

    retrieval_run = BenchmarkRunner(
        manifest,
        BenchmarkRunConfig(
            manifest_path="benchmarks/manifests/synthetic-smoke.json",
            suite="retrieval",
            mode="verified_lexical",
            output_dir=str(tmp_path / "drifted-retrieval"),
            pair_filters=("deleted",),
            seed_on_base_query_on_head=True,
        ),
    ).run()

    artifacts = benchmark_report.export_paper_tables(
        retrieval_locations=[str(Path(retrieval_run.artifacts["summary_json"]))],
        drift_locations=[],
        coding_locations=[],
        output_dir=tmp_path / "paper-tables",
    )

    retrieval_rows = list(
        csv.DictReader(Path(artifacts["retrieval_csv"]).read_text(encoding="utf-8").splitlines())
    )
    markdown = Path(artifacts["markdown"]).read_text(encoding="utf-8")

    assert retrieval_rows[0]["corpus"] == "Synthetic"
    assert retrieval_rows[0]["valid_recall_at_1"]
    assert retrieval_rows[0]["stale_rejection_rate"] == "1.0"
    assert retrieval_rows[0]["stale_hit_rate"] == "0.0"
    assert retrieval_rows[0]["safe_retrieval_at_1"] == "1.0"
    assert retrieval_rows[0]["safe_retrieval_at_5"] == "1.0"
    assert "| Corpus | Method | Cases |" in markdown
    assert "Valid Recall@1" in markdown
    assert "Valid Recall@5" in markdown
    assert "Stale Reject" in markdown
    assert "Stale Hit" in markdown
    assert "Safe@1" not in markdown
    assert "nDCG" not in markdown
    assert "Recall@1 | Recall@3 | Recall@5" not in markdown


def test_infer_corpus_label_maps_paper_run_names() -> None:
    cases = {
        "matrix-retrieval": "Synthetic",
        "matrix-retrieval-drifted-fixed": "Synthetic",
        "oss-click-flag-default": "Click",
        "oss-rich-cells-default": "Rich",
        "oss-attrs-frozen-default": "attrs",
        "controlled-coding-drift-super": "Controlled Coding Drift",
        "synthetic_controlled_coding_drift": "Controlled Coding Drift",
    }

    for run_label, expected in cases.items():
        run = benchmark_report.LoadedBenchmarkRun(
            summary_path=Path("/tmp") / run_label / "summary.json",
            run_label=run_label,
            result=BenchmarkRunResult(
                run_id=f"{run_label}-run",
                manifest_id="paper_manifest",
                manifest_digest="deadbeef",
                suite="retrieval",
                mode="verified_lexical",
                case_count=1,
                started_at="2026-01-01T00:00:00Z",
                completed_at="2026-01-01T00:00:01Z",
                metrics={},
                artifacts={},
            ),
        )
        assert benchmark_report._infer_corpus_label(run) == expected


def test_export_paper_tables_labels_controlled_coding_rows_from_path(
    tmp_path: Path,
) -> None:
    no_memory_summary = _write_summary(
        tmp_path / "controlled-coding-drift-super" / "no_memory" / "summary.json",
        suite="coding",
        mode="no_memory",
        manifest_id="paper_manifest",
        metrics={
            "pass_rate": 0.0,
            "resolved_rate": 0.0,
            "patch_applied_rate": 0.0,
            "retrieval_usage_rate": 0.0,
            "mean_verified_count": 0.0,
            "mean_stale_filtered_count": 0.0,
        },
    )
    naive_summary = _write_summary(
        tmp_path / "controlled-coding-drift-super" / "naive_lexical" / "summary.json",
        suite="coding",
        mode="naive_lexical",
        manifest_id="paper_manifest",
        metrics={
            "pass_rate": 1.0,
            "resolved_rate": 1.0,
            "patch_applied_rate": 1.0,
            "retrieval_usage_rate": 1.0,
            "mean_verified_count": 0.0,
            "mean_stale_filtered_count": 0.0,
        },
    )

    artifacts = benchmark_report.export_paper_tables(
        retrieval_locations=[],
        drift_locations=[],
        coding_locations=[str(no_memory_summary), str(naive_summary)],
        output_dir=tmp_path / "paper-tables",
    )

    coding_rows = list(
        csv.DictReader(Path(artifacts["coding_csv"]).read_text(encoding="utf-8").splitlines())
    )
    markdown = Path(artifacts["markdown"]).read_text(encoding="utf-8")
    metadata = json.loads(Path(artifacts["metadata_json"]).read_text(encoding="utf-8"))

    assert [row["corpus"] for row in coding_rows] == [
        "Controlled Coding Drift",
        "Controlled Coding Drift",
    ]
    assert [row["method"] for row in coding_rows] == ["No Memory", "Naive Lexical"]
    assert coding_rows[1]["corpus"] != coding_rows[1]["mode"]
    assert (
        "| Controlled Coding Drift | Naive Lexical | 1 | 1.000 | 1.000 | "
        "1.000 | 1.000 | 0.000 | 0.000 |"
    ) in markdown
    assert metadata["input_summary_json_paths"]["coding"] == [
        str(no_memory_summary),
        str(naive_summary),
    ]


def test_export_paper_tables_emits_metadata_json(tmp_path: Path) -> None:
    summary_path = _write_summary(
        tmp_path / "matrix-retrieval" / "runs" / "verified_lexical" / "summary.json",
        suite="retrieval",
        metrics={
            "recall_at_1": 1.0,
            "recall_at_3": 1.0,
            "recall_at_5": 1.0,
            "mrr": 1.0,
            "ndcg": 1.0,
            "median_latency_ms": 10.0,
        },
    )

    artifacts = benchmark_report.export_paper_tables(
        retrieval_locations=[str(summary_path)],
        drift_locations=[],
        coding_locations=[],
        output_dir=tmp_path / "paper-tables",
    )

    metadata_path = Path(artifacts["metadata_json"])
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert metadata_path.exists()
    assert metadata["generated_at"]
    assert metadata["input_summary_json_paths"] == {
        "retrieval": [str(summary_path)],
        "drift": [],
        "coding": [],
    }
    assert "git_branch" in metadata
    assert "git_commit_sha" in metadata


def test_export_paper_tables_resolves_runs_prefix_to_flat_coding_layout(
    tmp_path: Path,
) -> None:
    summary_path = _write_summary(
        tmp_path / "controlled-coding-drift-pilot" / "no_memory" / "summary.json",
        suite="coding",
        mode="no_memory",
        manifest_id="controlled_coding_drift",
        metrics={
            "pass_rate": 0.0,
            "resolved_rate": 0.0,
            "patch_applied_rate": 0.0,
            "retrieval_usage_rate": 0.0,
            "mean_verified_count": 0.0,
            "mean_stale_filtered_count": 0.0,
        },
    )

    artifacts = benchmark_report.export_paper_tables(
        retrieval_locations=[],
        drift_locations=[],
        coding_locations=[str(tmp_path / "controlled-coding-drift-pilot" / "runs" / "no_memory")],
        output_dir=tmp_path / "paper-tables",
    )

    coding_rows = list(
        csv.DictReader(Path(artifacts["coding_csv"]).read_text(encoding="utf-8").splitlines())
    )
    metadata = json.loads(Path(artifacts["metadata_json"]).read_text(encoding="utf-8"))

    assert coding_rows[0]["summary_json"] == str(summary_path)
    assert coding_rows[0]["corpus"] == "Controlled Coding Drift"
    assert metadata["input_summary_json_paths"]["coding"] == [str(summary_path)]


def test_benchmark_reporter_surfaces_attempt_deltas_and_memory_help() -> None:
    reporter = BenchmarkReporter()
    results = [
        BenchmarkCaseResult(
            suite="coding",
            mode="verified_lexical",
            case_id="repair-case",
            repo_name="synthetic_python_smoke",
            commit_pair_id="bugfix",
            metrics={
                "executed": True,
                "evaluated": True,
                "passed": True,
                "resolved": True,
                "testable": True,
                "patch_applied": True,
                "produced_patch_nonempty": True,
                    "retrieval_usage_rate": 1.0,
                    "verified_count": 2,
                    "relocated_count": 0,
                    "stale_filtered_count": 0,
                    "stale_hit_rate": 0.0,
                    "patch_similarity": 1.0,
                    "changed_path_f1": 1.0,
                    "context_chars": 100,
                    "runtime_ms": 25.0,
                    "tool_failure_count": 0,
            },
            metadata={"best_attempt_index": 2, "memory_context_count": 2},
        ),
        BenchmarkCaseResult(
            suite="coding",
            mode="verified_lexical",
            case_id="tool-case",
            repo_name="synthetic_python_smoke",
            commit_pair_id="bugfix",
            metrics={
                "executed": True,
                "evaluated": True,
                "passed": False,
                "resolved": False,
                "testable": True,
                "patch_applied": True,
                "produced_patch_nonempty": True,
                    "retrieval_usage_rate": 0.0,
                    "verified_count": 0,
                    "relocated_count": 0,
                    "stale_filtered_count": 0,
                    "stale_hit_rate": 0.0,
                    "patch_similarity": 0.5,
                    "changed_path_f1": 0.5,
                    "context_chars": 50,
                    "runtime_ms": 10.0,
                    "tool_failure_count": 1,
            },
            metadata={"best_attempt_index": 1, "memory_context_count": 0},
        ),
        BenchmarkCaseResult(
            suite="coding",
            mode="verified_lexical",
            case_id="verification-case",
            repo_name="synthetic_python_smoke",
            commit_pair_id="bugfix",
            metrics={
                "executed": True,
                "evaluated": True,
                "passed": False,
                "resolved": False,
                "testable": True,
                "patch_applied": True,
                "produced_patch_nonempty": True,
                    "retrieval_usage_rate": 0.0,
                    "verified_count": 0,
                    "relocated_count": 0,
                    "stale_filtered_count": 0,
                    "stale_hit_rate": 0.0,
                    "patch_similarity": 0.5,
                    "changed_path_f1": 0.5,
                    "context_chars": 50,
                    "runtime_ms": 12.0,
                    "tool_failure_count": 0,
            },
            metadata={"best_attempt_index": 1, "memory_context_count": 0},
        ),
        BenchmarkCaseResult(
            suite="coding",
            mode="verified_lexical",
            case_id="first-pass-case",
            repo_name="synthetic_python_smoke",
            commit_pair_id="bugfix",
            metrics={
                "executed": True,
                "evaluated": True,
                "passed": True,
                "resolved": True,
                "testable": True,
                "patch_applied": True,
                "produced_patch_nonempty": True,
                    "retrieval_usage_rate": 1.0,
                    "verified_count": 1,
                    "relocated_count": 0,
                    "stale_filtered_count": 0,
                    "stale_hit_rate": 0.0,
                    "patch_similarity": 1.0,
                    "changed_path_f1": 1.0,
                    "context_chars": 75,
                    "runtime_ms": 15.0,
                    "tool_failure_count": 0,
            },
            metadata={"best_attempt_index": 1, "memory_context_count": 1},
        ),
    ]
    attempts = [
        BenchmarkAttemptResult(
            suite="coding",
            mode="verified_lexical",
            case_id="repair-case",
            repo_name="synthetic_python_smoke",
            commit_pair_id="bugfix",
            attempt_index=1,
            metrics={
                "executed": True,
                "evaluated": True,
                "passed": False,
                "resolved": False,
                "testable": True,
                "patch_applied": False,
                "produced_patch_nonempty": False,
                "retrieval_usage_rate": 1.0,
                "context_chars": 80,
                "runtime_ms": 10.0,
                "tool_failure_count": 0,
            },
            metadata={"memory_context_count": 2, "produced_changed_paths": []},
        ),
        BenchmarkAttemptResult(
            suite="coding",
            mode="verified_lexical",
            case_id="repair-case",
            repo_name="synthetic_python_smoke",
            commit_pair_id="bugfix",
            attempt_index=2,
            metrics={
                "executed": True,
                "evaluated": True,
                "passed": True,
                "resolved": True,
                "testable": True,
                "patch_applied": True,
                "produced_patch_nonempty": True,
                "retrieval_usage_rate": 1.0,
                "context_chars": 110,
                "runtime_ms": 15.0,
                "tool_failure_count": 0,
            },
            metadata={"memory_context_count": 2, "produced_changed_paths": ["bugfix_module.py"]},
        ),
        BenchmarkAttemptResult(
            suite="coding",
            mode="verified_lexical",
            case_id="tool-case",
            repo_name="synthetic_python_smoke",
            commit_pair_id="bugfix",
            attempt_index=1,
            metrics={
                "executed": True,
                "evaluated": True,
                "passed": False,
                "resolved": False,
                "testable": True,
                "patch_applied": True,
                "produced_patch_nonempty": True,
                "retrieval_usage_rate": 0.0,
                "context_chars": 40,
                "runtime_ms": 10.0,
                "tool_failure_count": 1,
            },
            metadata={"memory_context_count": 0, "produced_changed_paths": ["tool_case.py"]},
        ),
        BenchmarkAttemptResult(
            suite="coding",
            mode="verified_lexical",
            case_id="verification-case",
            repo_name="synthetic_python_smoke",
            commit_pair_id="bugfix",
            attempt_index=1,
            metrics={
                "executed": True,
                "evaluated": True,
                "passed": False,
                "resolved": False,
                "testable": True,
                "patch_applied": True,
                "produced_patch_nonempty": True,
                "retrieval_usage_rate": 0.0,
                "context_chars": 40,
                "runtime_ms": 12.0,
                "tool_failure_count": 0,
            },
            metadata={
                "memory_context_count": 0,
                "produced_changed_paths": ["verification_case.py"],
            },
        ),
        BenchmarkAttemptResult(
            suite="coding",
            mode="verified_lexical",
            case_id="first-pass-case",
            repo_name="synthetic_python_smoke",
            commit_pair_id="bugfix",
            attempt_index=1,
            metrics={
                "executed": True,
                "evaluated": True,
                "passed": True,
                "resolved": True,
                "testable": True,
                "patch_applied": True,
                "produced_patch_nonempty": True,
                "retrieval_usage_rate": 1.0,
                "context_chars": 75,
                "runtime_ms": 15.0,
                "tool_failure_count": 0,
            },
            metadata={"memory_context_count": 1, "produced_changed_paths": ["first_pass.py"]},
        ),
    ]

    summary = reporter.summarize_results("coding", results, attempts=attempts, pass_k_values=(1, 2))

    assert summary["attempt_1_pass_rate"] == 0.25
    assert summary["attempt_2_rescue_rate"] == 1.0
    assert summary["attempt_failure_breakdown"]["empty_patch"] == 1
    assert summary["attempt_failure_breakdown"]["tool_failure"] == 1
    assert summary["attempt_failure_breakdown"]["verification"] == 1
    assert summary["memory_used_case_count"] == 2
    assert summary["memory_helped_case_count"] == 1
    assert summary["memory_helped_when_used_rate"] == 0.5
