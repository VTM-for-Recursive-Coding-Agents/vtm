from __future__ import annotations

import csv
from pathlib import Path

from vtm.benchmarks import BenchmarkManifest, BenchmarkRunConfig, BenchmarkRunner
from vtm.benchmarks import report as benchmark_report
from vtm.benchmarks.models import BenchmarkCaseResult, BenchmarkRunResult
from vtm.benchmarks.reporting import BenchmarkReporter
from vtm.enums import ValidityStatus


def _write_summary(
    path: Path,
    *,
    suite: str,
    mode: str = "verified_lexical",
    case_count: int = 1,
    metrics: dict[str, object] | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        BenchmarkRunResult(
            run_id=f"{path.parent.name}-run",
            manifest_id="synthetic_python_smoke",
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
    install_fake_vendored_rlm,
) -> None:
    install_fake_vendored_rlm()
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
            rlm_model_id="fake-model",
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

    assert retrieval_rows[0]["suite"] == "retrieval"
    assert retrieval_rows[0]["run_label"] == "retrieval"
    assert retrieval_rows[0]["method"] == "Verified Lexical"
    assert "recall_at_1" in retrieval_rows[0]
    assert drift_rows[0]["suite"] == "drift"
    assert drift_rows[0]["run_label"] == "drift"
    assert drift_rows[0]["status_confusion"]
    assert coding_rows[0]["suite"] == "coding"
    assert coding_rows[0]["run_label"] == "coding"
    assert coding_rows[0]["pass_at_1"] == "1.0"
    assert "## Retrieval" in markdown
    assert "## Drift" in markdown
    assert "## Coding" in markdown
    assert "| Run/Corpus | Method | Cases |" in markdown
    assert "| retrieval | Verified Lexical | 1 |" in markdown


def test_export_paper_tables_derives_run_label_from_matrix_layout(tmp_path: Path) -> None:
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
    assert "| matrix-drift | Verified Lexical | 1 | 1.000 | 1.000 | 0.000 | 12.0 |" in markdown


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

    assert retrieval_rows[0]["valid_recall_at_1"]
    assert retrieval_rows[0]["stale_rejection_rate"] == "1.0"
    assert retrieval_rows[0]["stale_hit_rate"] == "0.0"
    assert retrieval_rows[0]["safe_retrieval_at_1"] == "1.0"
    assert retrieval_rows[0]["safe_retrieval_at_5"] == "1.0"
    assert "Run/Corpus" in markdown
    assert "Valid Recall@1" in markdown
    assert "Valid Recall@5" in markdown
    assert "Stale Reject" in markdown
    assert "Safe@1" in markdown
    assert "Recall@1 | Recall@3 | Recall@5" not in markdown
