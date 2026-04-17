"""Dispatch layer that routes a benchmark run to the requested suite."""

from __future__ import annotations

from pathlib import Path
from typing import TypeVar

from vtm.adapters.rlm import RLMAdapter
from vtm.benchmarks.coding_suite import run_coding_suite
from vtm.benchmarks.drift_suite import run_drift_suite
from vtm.benchmarks.kernel_factory import BenchmarkKernelFactory
from vtm.benchmarks.models import (
    BenchmarkAttemptResult,
    BenchmarkCaseResult,
    BenchmarkManifest,
    BenchmarkRunConfig,
    CodingTaskCase,
    CommitPair,
    DriftCase,
    RepoSpec,
    RetrievalCase,
)
from vtm.benchmarks.repo_materialization import RepoWorkspaceManager
from vtm.benchmarks.retrieval_suite import run_retrieval_suite
from vtm.benchmarks.swebench_harness import SWEbenchHarnessRunner
from vtm.benchmarks.symbol_index import SymbolIndexer

CaseT = TypeVar("CaseT", RetrievalCase, DriftCase)


class BenchmarkSuiteExecutor:
    """Coordinates suite-specific runners using shared repo and kernel helpers."""

    def __init__(
        self,
        *,
        manifest: BenchmarkManifest,
        config: BenchmarkRunConfig,
        repo_manager: RepoWorkspaceManager,
        symbol_indexer: SymbolIndexer,
        rlm_adapter: RLMAdapter | None = None,
    ) -> None:
        """Create a suite executor with shared repo and indexing utilities."""
        self._manifest = manifest
        self._config = config
        self._repo_manager = repo_manager
        self._symbol_indexer = symbol_indexer
        self._kernel_factory = BenchmarkKernelFactory(
            config=config,
            symbol_indexer=symbol_indexer,
            rlm_adapter=rlm_adapter,
        )

    def run_retrieval_suite(
        self,
        output_dir: Path,
    ) -> tuple[list[RetrievalCase], list[BenchmarkCaseResult]]:
        """Execute the retrieval suite and apply any run-level case limit."""
        cases, results = run_retrieval_suite(
            output_dir=output_dir,
            selected_repo_pairs=self._iter_selected_repo_pairs(),
            repo_manager=self._repo_manager,
            symbol_indexer=self._symbol_indexer,
            kernel_factory=self._kernel_factory,
            top_k=self._config.top_k,
            mode=self._config.mode,
        )
        return self._limit_cases(cases, results)

    def run_drift_suite(
        self,
        output_dir: Path,
    ) -> tuple[list[DriftCase], list[BenchmarkCaseResult]]:
        """Execute the drift suite and apply any run-level case limit."""
        cases, results = run_drift_suite(
            output_dir=output_dir,
            selected_repo_pairs=self._iter_selected_repo_pairs(),
            repo_manager=self._repo_manager,
            symbol_indexer=self._symbol_indexer,
            kernel_factory=self._kernel_factory,
            mode=self._config.mode,
        )
        return self._limit_cases(cases, results)

    def run_coding_suite(
        self,
        output_dir: Path,
    ) -> tuple[list[CodingTaskCase], list[BenchmarkCaseResult], list[BenchmarkAttemptResult]]:
        """Execute the coding suite."""
        return run_coding_suite(
            output_dir=output_dir,
            manifest=self._manifest,
            config=self._config,
            selected_repo_pairs=self._iter_selected_repo_pairs(),
            repo_manager=self._repo_manager,
            symbol_indexer=self._symbol_indexer,
            kernel_factory=self._kernel_factory,
            require_pair=self._require_pair,
            swebench_harness_runner=self._swebench_harness_runner(output_dir),
        )

    def _limit_cases(
        self,
        cases: list[CaseT],
        results: list[BenchmarkCaseResult],
    ) -> tuple[list[CaseT], list[BenchmarkCaseResult]]:
        if self._config.max_cases is None:
            return cases, results
        selected_cases = cases[: self._config.max_cases]
        result_map = {result.case_id: result for result in results}
        return selected_cases, [result_map[case.case_id] for case in selected_cases]

    def _iter_selected_repo_pairs(self) -> list[tuple[RepoSpec, CommitPair]]:
        selected_repo_names = set(self._config.repo_filters)
        selected_pair_ids = set(self._config.pair_filters)
        matched_repo_names: set[str] = set()
        matched_pair_ids: set[str] = set()
        selected: list[tuple[RepoSpec, CommitPair]] = []

        for repo_spec in self._manifest.repos:
            if selected_repo_names and repo_spec.repo_name not in selected_repo_names:
                continue
            matched_repo_names.add(repo_spec.repo_name)
            for pair in repo_spec.commit_pairs:
                if selected_pair_ids and pair.pair_id not in selected_pair_ids:
                    continue
                matched_pair_ids.add(pair.pair_id)
                selected.append((repo_spec, pair))

        missing_repos = sorted(selected_repo_names - matched_repo_names)
        if missing_repos:
            raise ValueError(f"unknown benchmark repos: {missing_repos}")
        missing_pairs = sorted(selected_pair_ids - matched_pair_ids)
        if missing_pairs:
            raise ValueError(f"unknown benchmark pairs: {missing_pairs}")
        return selected

    def _require_pair(self, repo_spec: RepoSpec, pair_id: str) -> CommitPair:
        for pair in repo_spec.commit_pairs:
            if pair.pair_id == pair_id:
                return pair
        raise KeyError(f"unknown commit pair: {repo_spec.repo_name}:{pair_id}")

    def _swebench_harness_runner(self, output_dir: Path) -> SWEbenchHarnessRunner:
        del output_dir
        return SWEbenchHarnessRunner()


__all__ = ["BenchmarkSuiteExecutor"]
