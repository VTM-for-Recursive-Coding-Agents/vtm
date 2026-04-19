"""CLI helpers for exporting paper-ready benchmark summary tables."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vtm.base import utc_now
from vtm.benchmarks.models import BenchmarkMode, BenchmarkRunResult, BenchmarkSuite

MODE_ORDER: dict[BenchmarkMode, int] = {
    "no_memory": 0,
    "naive_lexical": 1,
    "verified_lexical": 2,
    "lexical_rlm_rerank": 3,
}

MODE_LABELS: dict[BenchmarkMode, str] = {
    "no_memory": "No Memory",
    "naive_lexical": "Naive Lexical",
    "verified_lexical": "Verified Lexical",
    "lexical_rlm_rerank": "Lexical + RLM Rerank",
}

CSV_BASE_FIELDS = (
    "corpus",
    "run_label",
    "suite",
    "mode",
    "method",
    "run_id",
    "manifest_id",
    "case_count",
    "summary_json",
)

RETRIEVAL_METRIC_ORDER = (
    "recall_at_1",
    "recall_at_3",
    "recall_at_5",
    "valid_recall_at_1",
    "valid_recall_at_3",
    "valid_recall_at_5",
    "stale_rejection_rate",
    "stale_hit_rate",
    "safe_retrieval_at_1",
    "safe_retrieval_at_5",
    "mrr",
    "ndcg",
    "median_latency_ms",
    "mean_verified_count",
    "mean_relocated_count",
    "mean_stale_filtered_count",
    "mean_stale_hit_rate",
)

DRIFT_METRIC_ORDER = (
    "relocation_precision",
    "relocation_recall",
    "false_verified_rate",
    "median_verification_latency_ms",
    "status_confusion",
)

CODING_METRIC_ORDER = (
    "pass_rate",
    "resolved_rate",
    "patch_applied_rate",
    "pass_at_1",
    "median_runtime_ms",
    "retrieval_usage_rate",
    "mean_verified_count",
    "mean_relocated_count",
    "mean_stale_filtered_count",
    "mean_stale_hit_rate",
)


@dataclass(frozen=True)
class LoadedBenchmarkRun:
    """Completed benchmark run plus the resolved on-disk summary path."""

    summary_path: Path
    run_label: str
    result: BenchmarkRunResult


def build_parser() -> argparse.ArgumentParser:
    """Build the paper table export CLI parser."""
    parser = argparse.ArgumentParser(
        description="Export deterministic CSV and Markdown paper tables from completed runs."
    )
    parser.add_argument(
        "--retrieval-run",
        action="append",
        default=[],
        help="Completed retrieval run directory or summary.json path. Repeat to add rows.",
    )
    parser.add_argument(
        "--drift-run",
        action="append",
        default=[],
        help="Completed drift run directory or summary.json path. Repeat to add rows.",
    )
    parser.add_argument(
        "--coding-run",
        action="append",
        default=[],
        help="Completed coding run directory or summary.json path. Repeat to add rows.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Directory for exported paper tables.",
    )
    return parser


def main() -> int:
    """Export suite summaries and print the generated artifact locations as JSON."""
    args = build_parser().parse_args()
    artifacts = export_paper_tables(
        retrieval_locations=args.retrieval_run,
        drift_locations=args.drift_run,
        coding_locations=args.coding_run,
        output_dir=args.output,
    )
    print(json.dumps(artifacts, indent=2, sort_keys=True))
    return 0


def export_paper_tables(
    *,
    retrieval_locations: list[str],
    drift_locations: list[str],
    coding_locations: list[str],
    output_dir: str | Path,
) -> dict[str, str]:
    """Load completed benchmark runs and export suite CSVs plus combined Markdown."""
    if not retrieval_locations and not drift_locations and not coding_locations:
        raise ValueError("provide at least one --retrieval-run, --drift-run, or --coding-run")

    retrieval_runs = _load_runs(retrieval_locations, expected_suite="retrieval")
    drift_runs = _load_runs(drift_locations, expected_suite="drift")
    coding_runs = _load_runs(coding_locations, expected_suite="coding")

    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, str] = {}

    if retrieval_runs:
        retrieval_csv_path = resolved_output_dir / "retrieval_summary.csv"
        _write_csv(
            retrieval_csv_path,
            *_suite_csv_rows(retrieval_runs, preferred_metric_order=RETRIEVAL_METRIC_ORDER),
        )
        artifacts["retrieval_csv"] = str(retrieval_csv_path)
    if drift_runs:
        drift_csv_path = resolved_output_dir / "drift_summary.csv"
        _write_csv(
            drift_csv_path,
            *_suite_csv_rows(drift_runs, preferred_metric_order=DRIFT_METRIC_ORDER),
        )
        artifacts["drift_csv"] = str(drift_csv_path)
    if coding_runs:
        coding_csv_path = resolved_output_dir / "coding_summary.csv"
        _write_csv(
            coding_csv_path,
            *_suite_csv_rows(coding_runs, preferred_metric_order=CODING_METRIC_ORDER),
        )
        artifacts["coding_csv"] = str(coding_csv_path)

    markdown_path = resolved_output_dir / "paper_tables.md"
    markdown_path.write_text(
        _render_markdown(
            retrieval_runs=retrieval_runs,
            drift_runs=drift_runs,
            coding_runs=coding_runs,
        ),
        encoding="utf-8",
    )
    artifacts["markdown"] = str(markdown_path)

    metadata_path = resolved_output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            _build_metadata(
                retrieval_runs=retrieval_runs,
                drift_runs=drift_runs,
                coding_runs=coding_runs,
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    artifacts["metadata_json"] = str(metadata_path)
    return artifacts


def _load_runs(
    locations: list[str],
    *,
    expected_suite: BenchmarkSuite,
) -> list[LoadedBenchmarkRun]:
    runs: list[LoadedBenchmarkRun] = []
    for location in locations:
        summary_path = _resolve_summary_path(Path(location))
        result = BenchmarkRunResult.from_json(summary_path.read_text(encoding="utf-8"))
        if result.suite != expected_suite:
            raise ValueError(
                f"expected {expected_suite} run but found {result.suite}: {summary_path}"
            )
        runs.append(
            LoadedBenchmarkRun(
                summary_path=summary_path,
                run_label=_derive_run_label(summary_path),
                result=result,
            )
        )
    return sorted(
        runs,
        key=lambda run: (
            MODE_ORDER[run.result.mode],
            run.run_label,
            run.result.manifest_id,
            run.result.run_id,
            str(run.summary_path),
        ),
    )


def _resolve_summary_path(location: Path) -> Path:
    summary_path = location / "summary.json" if location.is_dir() else location
    if not summary_path.exists():
        alternate = _alternate_summary_path(location)
        if alternate is not None and alternate.exists():
            return alternate
    if not summary_path.exists():
        raise FileNotFoundError(f"benchmark summary path does not exist: {summary_path}")
    return summary_path


def _alternate_summary_path(location: Path) -> Path | None:
    normalized = Path(str(location).rstrip("/"))
    parts = normalized.parts
    if len(parts) >= 2 and parts[-2] == "runs":
        collapsed = Path(*parts[:-2], parts[-1])
        return collapsed / "summary.json"
    return None


def _derive_run_label(summary_path: Path) -> str:
    run_dir = summary_path.parent
    for ancestor in (run_dir, *run_dir.parents):
        if ancestor.name == "runs" and ancestor.parent.name:
            return ancestor.parent.name
    return run_dir.name or summary_path.stem


def _suite_csv_rows(
    runs: list[LoadedBenchmarkRun],
    *,
    preferred_metric_order: tuple[str, ...],
) -> tuple[list[str], list[dict[str, Any]]]:
    metric_names = _ordered_metric_names(runs, preferred_metric_order=preferred_metric_order)
    fieldnames = [*CSV_BASE_FIELDS, *metric_names]
    rows: list[dict[str, Any]] = []
    for run in runs:
        row: dict[str, Any] = {
            "corpus": _infer_corpus_label(run),
            "run_label": run.run_label,
            "suite": run.result.suite,
            "mode": run.result.mode,
            "method": MODE_LABELS[run.result.mode],
            "run_id": run.result.run_id,
            "manifest_id": run.result.manifest_id,
            "case_count": run.result.case_count,
            "summary_json": str(run.summary_path),
        }
        for metric_name in metric_names:
            row[metric_name] = _serialize_csv_value(run.result.metrics.get(metric_name))
        rows.append(row)
    return fieldnames, rows


def _ordered_metric_names(
    runs: list[LoadedBenchmarkRun],
    *,
    preferred_metric_order: tuple[str, ...],
) -> list[str]:
    present_metrics = {
        metric_name
        for run in runs
        for metric_name in run.result.metrics
    }
    ordered = [name for name in preferred_metric_order if name in present_metrics]
    extras = sorted(present_metrics.difference(preferred_metric_order))
    return [*ordered, *extras]


def _serialize_csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return value


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _render_markdown(
    *,
    retrieval_runs: list[LoadedBenchmarkRun],
    drift_runs: list[LoadedBenchmarkRun],
    coding_runs: list[LoadedBenchmarkRun],
) -> str:
    lines = ["# VTM Paper Tables", ""]
    if retrieval_runs:
        include_drifted_metrics = any(
            any(
                run.result.metrics.get(metric_name) is not None
                for metric_name in (
                    "valid_recall_at_1",
                    "valid_recall_at_3",
                    "valid_recall_at_5",
                    "stale_rejection_rate",
                    "stale_hit_rate",
                    "safe_retrieval_at_1",
                )
            )
            for run in retrieval_runs
        )
        lines.extend(["## Retrieval", ""])
        if include_drifted_metrics:
            lines.extend(
                [
                    (
                        "| Corpus | Method | Cases | Valid Recall@1 | "
                        "Valid Recall@5 | Stale Reject | Stale Hit | MRR |"
                    ),
                    (
                        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"
                    ),
                ]
            )
        else:
            lines.extend(
                [
                    "| Corpus | Method | Cases | Recall@1 | Recall@3 | Recall@5 | MRR | nDCG |",
                    "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
                ]
            )
        for run in retrieval_runs:
            metrics = run.result.metrics
            if include_drifted_metrics:
                lines.append(
                    "| "
                    f"{_infer_corpus_label(run)} | "
                    f"{MODE_LABELS[run.result.mode]} | "
                    f"{run.result.case_count} | "
                    f"{_format_metric(metrics.get('valid_recall_at_1'))} | "
                    f"{_format_metric(metrics.get('valid_recall_at_5'))} | "
                    f"{_format_metric(metrics.get('stale_rejection_rate'))} | "
                    f"{_format_metric(metrics.get('stale_hit_rate'))} | "
                    f"{_format_metric(metrics.get('mrr'))} |"
                )
            else:
                lines.append(
                    "| "
                    f"{_infer_corpus_label(run)} | "
                    f"{MODE_LABELS[run.result.mode]} | "
                    f"{run.result.case_count} | "
                    f"{_format_metric(metrics.get('recall_at_1'))} | "
                    f"{_format_metric(metrics.get('recall_at_3'))} | "
                    f"{_format_metric(metrics.get('recall_at_5'))} | "
                    f"{_format_metric(metrics.get('mrr'))} | "
                    f"{_format_metric(metrics.get('ndcg'))} |"
                )
        lines.append("")
    if drift_runs:
        lines.extend(
            [
                "## Drift",
                "",
                (
                    "| Corpus | Cases | Relocation Precision | Relocation Recall | "
                    "False Verified Rate | Median Verification Latency (ms) |"
                ),
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for run in drift_runs:
            metrics = run.result.metrics
            lines.append(
                "| "
                f"{_infer_corpus_label(run)} | "
                f"{run.result.case_count} | "
                f"{_format_metric(metrics.get('relocation_precision'))} | "
                f"{_format_metric(metrics.get('relocation_recall'))} | "
                f"{_format_metric(metrics.get('false_verified_rate'))} | "
                f"{_format_metric(metrics.get('median_verification_latency_ms'), digits=1)} |"
            )
        lines.append("")
    if coding_runs:
        lines.extend(
            [
                "## Coding",
                "",
                (
                    "| Corpus | Method | Cases | Pass Rate | Resolved Rate | "
                    "Patch Applied Rate | Retrieval Usage Rate | Mean Verified Count | "
                    "Mean Stale Filtered Count |"
                ),
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for run in coding_runs:
            metrics = run.result.metrics
            lines.append(
                "| "
                f"{_infer_corpus_label(run)} | "
                f"{MODE_LABELS[run.result.mode]} | "
                f"{run.result.case_count} | "
                f"{_format_metric(metrics.get('pass_rate'))} | "
                f"{_format_metric(metrics.get('resolved_rate'))} | "
                f"{_format_metric(metrics.get('patch_applied_rate'))} | "
                f"{_format_metric(metrics.get('retrieval_usage_rate'))} | "
                f"{_format_metric(metrics.get('mean_verified_count'))} | "
                f"{_format_metric(metrics.get('mean_stale_filtered_count'))} |"
            )
        lines.append("")
    return "\n".join(lines)


def _infer_corpus_label(run: LoadedBenchmarkRun) -> str:
    candidates = _corpus_candidates(run)
    for candidate in candidates:
        if candidate in {
            "matrix-drift",
            "matrix-retrieval",
            "matrix-retrieval-drifted-fixed",
        }:
            return "Synthetic"
        if candidate.startswith("oss-click-"):
            return "Click"
        if candidate.startswith("oss-rich-"):
            return "Rich"
        if candidate.startswith("oss-attrs-"):
            return "attrs"
        if "controlled-coding-drift" in candidate:
            return "Controlled Coding Drift"
        if "synthetic" in candidate:
            return "Synthetic"
    return run.run_label


def _corpus_candidates(run: LoadedBenchmarkRun) -> tuple[str, ...]:
    candidates: list[str] = [
        _normalize_label(run.run_label),
        _normalize_label(run.result.manifest_id),
    ]
    for path in (run.summary_path.parent, *run.summary_path.parents):
        if path.name:
            candidates.append(_normalize_label(path.name))
    return tuple(dict.fromkeys(candidates))


def _normalize_label(value: str) -> str:
    return value.strip().lower().replace("_", "-")


def _build_metadata(
    *,
    retrieval_runs: list[LoadedBenchmarkRun],
    drift_runs: list[LoadedBenchmarkRun],
    coding_runs: list[LoadedBenchmarkRun],
) -> dict[str, Any]:
    return {
        "git_branch": _git_command_output("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "git_commit_sha": _git_command_output("git", "rev-parse", "HEAD"),
        "generated_at": utc_now().isoformat(),
        "input_summary_json_paths": {
            "retrieval": [str(run.summary_path) for run in retrieval_runs],
            "drift": [str(run.summary_path) for run in drift_runs],
            "coding": [str(run.summary_path) for run in coding_runs],
        },
    }


def _git_command_output(*args: str) -> str | None:
    try:
        completed = subprocess.run(
            args,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def _format_metric(value: Any, *, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
