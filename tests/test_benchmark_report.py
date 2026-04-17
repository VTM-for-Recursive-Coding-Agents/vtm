from __future__ import annotations

import csv
from pathlib import Path

from vtm.benchmarks import BenchmarkManifest, BenchmarkRunConfig, BenchmarkRunner
from vtm.benchmarks import report as benchmark_report


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
    assert retrieval_rows[0]["method"] == "Verified Lexical"
    assert "recall_at_1" in retrieval_rows[0]
    assert drift_rows[0]["suite"] == "drift"
    assert drift_rows[0]["status_confusion"]
    assert coding_rows[0]["suite"] == "coding"
    assert coding_rows[0]["pass_at_1"] == "1.0"
    assert "## Retrieval" in markdown
    assert "## Drift" in markdown
    assert "## Coding" in markdown
    assert "| Verified Lexical | 1 |" in markdown
