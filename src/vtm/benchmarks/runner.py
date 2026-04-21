"""Top-level benchmark runner that writes cases, results, and summaries."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

from vtm.base import utc_now
from vtm.benchmarks.models import (
    BenchmarkAttemptResult,
    BenchmarkCaseResult,
    BenchmarkManifest,
    BenchmarkRunConfig,
    BenchmarkRunResult,
    CodingTaskCase,
    DriftCase,
    RetrievalCase,
)
from vtm.benchmarks.repo_materialization import RepoWorkspaceManager
from vtm.benchmarks.reporting import BenchmarkReporter
from vtm.benchmarks.suite_execution import BenchmarkSuiteExecutor
from vtm.benchmarks.symbol_index import SymbolIndexer


class BenchmarkRunner:
    """Orchestrates one benchmark run from manifest load to summary artifacts."""

    def __init__(
        self,
        manifest: BenchmarkManifest,
        config: BenchmarkRunConfig,
    ) -> None:
        """Bind the runner to a manifest and config."""
        self._manifest = manifest
        self._config = config
        self._executor = BenchmarkSuiteExecutor(
            manifest=manifest,
            config=config,
            repo_manager=RepoWorkspaceManager(),
            symbol_indexer=SymbolIndexer(),
        )
        self._reporter = BenchmarkReporter()

    def run(self) -> BenchmarkRunResult:
        """Execute the configured suite and persist run artifacts."""
        started_at = utc_now()
        output_dir = Path(self._config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_lock: dict[str, object] = {
            "manifest": self._manifest.model_dump(mode="json"),
            "suite": self._config.suite,
            "mode": self._config.mode,
            "seed": self._config.seed,
        }
        if self._config.seed_on_base_query_on_head:
            manifest_lock["seed_on_base_query_on_head"] = True
        if self._config.suite == "coding":
            manifest_lock["execution_engine"] = self._config.coding_engine
            manifest_lock["workspace_backend"] = self._config.workspace_backend
            manifest_lock["attempt_count"] = self._config.attempt_count
            manifest_lock["pass_k_values"] = list(self._config.pass_k_values)
            manifest_lock["execution_model_id"] = self._config.execution_model_id
            manifest_lock["agent_max_iterations"] = self._config.agent_max_iterations
            manifest_lock["workspace_command_timeout_seconds"] = (
                self._config.workspace_command_timeout_seconds
            )
            manifest_lock["workspace_max_output_chars"] = (
                self._config.workspace_max_output_chars
            )
            if self._config.workspace_backend == "docker_workspace":
                manifest_lock["docker_image"] = self._config.docker_image
                manifest_lock["docker_binary"] = self._config.docker_binary
                manifest_lock["docker_network"] = self._config.docker_network
        if self._config.repo_filters:
            manifest_lock["repo_filters"] = list(self._config.repo_filters)
        if self._config.pair_filters:
            manifest_lock["pair_filters"] = list(self._config.pair_filters)
        manifest_digest = hashlib.sha256(
            json.dumps(manifest_lock, separators=(",", ":"), sort_keys=True).encode("utf-8")
        ).hexdigest()
        run_id = manifest_digest[:16]
        cases: Sequence[RetrievalCase | DriftCase | CodingTaskCase]
        results: list[BenchmarkCaseResult]
        attempt_results: list[BenchmarkAttemptResult] = []

        if self._config.suite == "retrieval":
            cases, results = self._executor.run_retrieval_suite(output_dir)
        elif self._config.suite == "drift":
            cases, results = self._executor.run_drift_suite(output_dir)
        else:
            cases, results, attempt_results = self._executor.run_coding_suite(output_dir)

        self._validate_case_results_alignment(cases, results)
        summary_metrics = self._reporter.summarize_results(
            self._config.suite,
            results,
            attempts=attempt_results,
            pass_k_values=self._config.pass_k_values,
        )
        manifest_lock_path = output_dir / "manifest.lock.json"
        cases_path = output_dir / "cases.jsonl"
        results_path = output_dir / "results.jsonl"
        attempts_path = output_dir / "attempts.jsonl"
        summary_json_path = output_dir / "summary.json"
        summary_md_path = output_dir / "summary.md"

        manifest_lock_path.write_text(
            json.dumps(manifest_lock, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        self._reporter.write_jsonl(cases_path, [case.model_dump(mode="json") for case in cases])
        self._reporter.write_jsonl(
            results_path,
            [result.model_dump(mode="json") for result in results],
        )
        if attempt_results:
            self._reporter.write_jsonl(
                attempts_path,
                [attempt.model_dump(mode="json") for attempt in attempt_results],
            )

        run_result = BenchmarkRunResult(
            run_id=run_id,
            manifest_id=self._manifest.manifest_id,
            manifest_digest=manifest_digest,
            suite=self._config.suite,
            mode=self._config.mode,
            case_count=len(cases),
            started_at=started_at.isoformat(),
            completed_at=utc_now().isoformat(),
            metrics=summary_metrics,
            artifacts={
                "manifest_lock": str(manifest_lock_path),
                "cases_jsonl": str(cases_path),
                "results_jsonl": str(results_path),
                "summary_json": str(summary_json_path),
                "summary_md": str(summary_md_path),
            },
        )
        if attempt_results:
            run_result.artifacts["attempts_jsonl"] = str(attempts_path)
        summary_json_path.write_text(run_result.to_json(), encoding="utf-8")
        summary_md_path.write_text(self._reporter.render_summary(run_result), encoding="utf-8")
        return run_result

    def _validate_case_results_alignment(
        self,
        cases: Sequence[RetrievalCase | DriftCase | CodingTaskCase],
        results: Sequence[BenchmarkCaseResult],
    ) -> None:
        """Reject mismatches between emitted case IDs and result IDs."""
        case_ids = [case.case_id for case in cases]
        result_ids = [result.case_id for result in results]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("benchmark cases must have unique case_id values")
        if len(result_ids) != len(set(result_ids)):
            raise ValueError("benchmark results must have unique case_id values")
        if len(case_ids) != len(result_ids):
            raise ValueError(
                "benchmark case/result count mismatch: "
                f"{len(case_ids)} cases vs {len(result_ids)} results"
            )
        if set(case_ids) != set(result_ids):
            missing_results = sorted(set(case_ids) - set(result_ids))
            unexpected_results = sorted(set(result_ids) - set(case_ids))
            raise ValueError(
                "benchmark case/result IDs diverged: "
                f"missing={missing_results} unexpected={unexpected_results}"
            )
