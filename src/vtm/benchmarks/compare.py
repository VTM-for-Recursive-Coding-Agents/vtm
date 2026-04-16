"""CLI entrypoint for comparing two completed benchmark runs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

from vtm.benchmarks.models import (
    BenchmarkAttemptResult,
    BenchmarkCaseResult,
    BenchmarkComparisonResult,
    BenchmarkRunResult,
)
from vtm.benchmarks.reporting import BenchmarkReporter


def build_parser() -> argparse.ArgumentParser:
    """Build the benchmark comparison CLI parser."""
    parser = argparse.ArgumentParser(description="Compare two completed VTM benchmark runs.")
    parser.add_argument(
        "--baseline",
        required=True,
        help="Baseline run directory or summary.json path.",
    )
    parser.add_argument(
        "--candidate",
        required=True,
        help="Candidate run directory or summary.json path.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Directory for comparison outputs.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=2000,
        help="Bootstrap samples used for paired numeric confidence intervals.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=0,
        help="Deterministic seed used for bootstrap resampling.",
    )
    return parser


def main() -> int:
    """Load two completed runs, compare them, and write comparison artifacts."""
    args = build_parser().parse_args()
    comparison = compare_completed_runs(
        baseline_location=args.baseline,
        candidate_location=args.candidate,
        output_dir=args.output,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(comparison.to_json())
    return 0


def compare_completed_runs(
    *,
    baseline_location: str | Path,
    candidate_location: str | Path,
    output_dir: str | Path,
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 0,
) -> BenchmarkComparisonResult:
    """Compare two completed run directories and write durable comparison artifacts."""
    reporter = BenchmarkReporter()
    baseline, baseline_dir, baseline_results, baseline_attempts = _load_run_bundle(
        str(baseline_location)
    )
    candidate, candidate_dir, candidate_results, candidate_attempts = _load_run_bundle(
        str(candidate_location)
    )

    comparison = reporter.compare_runs(
        baseline=baseline,
        candidate=candidate,
        baseline_results=baseline_results,
        candidate_results=candidate_results,
        baseline_attempts=baseline_attempts,
        candidate_attempts=candidate_attempts,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )

    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    comparison_json_path = resolved_output_dir / "comparison.json"
    comparison_md_path = resolved_output_dir / "comparison.md"
    artifacts = {
        "baseline_summary_json": str(
            _artifact_path(baseline_dir, baseline.artifacts["summary_json"])
        ),
        "baseline_results_jsonl": str(
            _artifact_path(baseline_dir, baseline.artifacts["results_jsonl"])
        ),
        "candidate_summary_json": str(
            _artifact_path(candidate_dir, candidate.artifacts["summary_json"])
        ),
        "candidate_results_jsonl": str(
            _artifact_path(candidate_dir, candidate.artifacts["results_jsonl"])
        ),
        "comparison_json": str(comparison_json_path),
        "comparison_md": str(comparison_md_path),
    }
    if "attempts_jsonl" in baseline.artifacts:
        artifacts["baseline_attempts_jsonl"] = str(
            _artifact_path(baseline_dir, baseline.artifacts["attempts_jsonl"])
        )
    if "attempts_jsonl" in candidate.artifacts:
        artifacts["candidate_attempts_jsonl"] = str(
            _artifact_path(candidate_dir, candidate.artifacts["attempts_jsonl"])
        )
    comparison = comparison.model_copy(update={"artifacts": artifacts})

    comparison_json_path.write_text(comparison.to_json(), encoding="utf-8")
    comparison_md_path.write_text(reporter.render_comparison(comparison), encoding="utf-8")
    return comparison


def _load_run_bundle(
    location: str,
) -> tuple[
    BenchmarkRunResult,
    Path,
    list[BenchmarkCaseResult],
    list[BenchmarkAttemptResult],
]:
    summary_path = _resolve_summary_path(Path(location))
    run_dir = summary_path.parent
    run_result = BenchmarkRunResult.from_json(summary_path.read_text(encoding="utf-8"))
    results_path = _artifact_path(run_dir, run_result.artifacts["results_jsonl"])
    attempts_path = (
        _artifact_path(run_dir, run_result.artifacts["attempts_jsonl"])
        if "attempts_jsonl" in run_result.artifacts
        else None
    )
    return (
        run_result,
        run_dir,
        _load_jsonl_models(results_path, BenchmarkCaseResult),
        []
        if attempts_path is None
        else _load_jsonl_models(attempts_path, BenchmarkAttemptResult),
    )


def _resolve_summary_path(location: Path) -> Path:
    if location.is_dir():
        summary_path = location / "summary.json"
    else:
        summary_path = location
    if not summary_path.exists():
        raise FileNotFoundError(f"benchmark summary path does not exist: {summary_path}")
    return summary_path


def _artifact_path(run_dir: Path, artifact: str) -> Path:
    path = Path(artifact)
    if path.is_absolute():
        return path
    return run_dir / path


def _load_jsonl_models[ModelT: BenchmarkCaseResult | BenchmarkAttemptResult](
    path: Path,
    model: type[ModelT],
) -> list[ModelT]:
    rows = path.read_text(encoding="utf-8").splitlines()
    loaded: list[ModelT] = []
    for row in rows:
        if row.strip():
            loaded.append(cast(ModelT, model.model_validate_json(row)))
    return loaded


if __name__ == "__main__":
    raise SystemExit(main())
