from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

from vtm.adapters.embeddings import EmbeddingAdapter
from vtm.adapters.rlm import RLMAdapter
from vtm.base import utc_now
from vtm.benchmarks.models import (
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
    def __init__(
        self,
        manifest: BenchmarkManifest,
        config: BenchmarkRunConfig,
        *,
        rlm_adapter: RLMAdapter | None = None,
        embedding_adapter: EmbeddingAdapter | None = None,
    ) -> None:
        self._manifest = manifest
        self._config = config
        self._embedding_adapter = embedding_adapter
        self._executor = BenchmarkSuiteExecutor(
            manifest=manifest,
            config=config,
            repo_manager=RepoWorkspaceManager(),
            symbol_indexer=SymbolIndexer(),
            rlm_adapter=rlm_adapter,
            embedding_adapter=embedding_adapter,
        )
        self._reporter = BenchmarkReporter()

    def run(self) -> BenchmarkRunResult:
        started_at = utc_now()
        output_dir = Path(self._config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_lock: dict[str, object] = {
            "manifest": self._manifest.model_dump(mode="json"),
            "suite": self._config.suite,
            "mode": self._config.mode,
            "seed": self._config.seed,
        }
        if self._config.repo_filters:
            manifest_lock["repo_filters"] = list(self._config.repo_filters)
        if self._config.pair_filters:
            manifest_lock["pair_filters"] = list(self._config.pair_filters)
        if self._config.swebench_dataset_name:
            manifest_lock["swebench_dataset_name"] = self._config.swebench_dataset_name
        manifest_lock["swebench_harness_workers"] = self._config.swebench_harness_workers
        manifest_lock["swebench_harness_cache_level"] = self._config.swebench_harness_cache_level
        if self._config.swebench_harness_run_id:
            manifest_lock["swebench_harness_run_id"] = self._config.swebench_harness_run_id
        if self._config.mode == "embedding":
            manifest_lock["embedding_adapter"] = (
                getattr(self._embedding_adapter, "adapter_id", None)
                or "deterministic_hash:64"
            )
        manifest_digest = hashlib.sha256(
            json.dumps(manifest_lock, separators=(",", ":"), sort_keys=True).encode("utf-8")
        ).hexdigest()
        run_id = manifest_digest[:16]
        cases: Sequence[RetrievalCase | DriftCase | CodingTaskCase]
        results: list[BenchmarkCaseResult]

        if self._config.suite == "retrieval":
            cases, results = self._executor.run_retrieval_suite(output_dir)
        elif self._config.suite == "drift":
            cases, results = self._executor.run_drift_suite(output_dir)
        else:
            cases, results = self._executor.run_coding_suite(output_dir)

        self._validate_case_results_alignment(cases, results)
        summary_metrics = self._reporter.summarize_results(self._config.suite, results)
        manifest_lock_path = output_dir / "manifest.lock.json"
        cases_path = output_dir / "cases.jsonl"
        results_path = output_dir / "results.jsonl"
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
        for name in ("predictions.jsonl", "swebench_harness_results.json"):
            candidate = output_dir / name
            if candidate.exists():
                artifact_key = (
                    "predictions_jsonl"
                    if name.endswith(".jsonl")
                    else "swebench_harness_results_json"
                )
                run_result.artifacts[artifact_key] = str(candidate)
        logs_dir = output_dir / "logs"
        if logs_dir.exists():
            run_result.artifacts["swebench_harness_logs_dir"] = str(logs_dir)
        summary_json_path.write_text(run_result.to_json(), encoding="utf-8")
        summary_md_path.write_text(self._reporter.render_summary(run_result), encoding="utf-8")
        return run_result

    def _validate_case_results_alignment(
        self,
        cases: Sequence[RetrievalCase | DriftCase | CodingTaskCase],
        results: Sequence[BenchmarkCaseResult],
    ) -> None:
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
