"""Drift benchmark suite execution helpers."""

from __future__ import annotations

import time
from pathlib import Path

from vtm.benchmarks.kernel_factory import BenchmarkKernelFactory
from vtm.benchmarks.models import (
    BenchmarkCaseResult,
    BenchmarkMode,
    CommitPair,
    DriftCase,
    RepoSpec,
)
from vtm.benchmarks.repo_materialization import RepoWorkspaceManager
from vtm.benchmarks.symbol_index import SymbolIndexer


def run_drift_suite(
    *,
    output_dir: Path,
    selected_repo_pairs: list[tuple[RepoSpec, CommitPair]],
    repo_manager: RepoWorkspaceManager,
    symbol_indexer: SymbolIndexer,
    kernel_factory: BenchmarkKernelFactory,
    mode: BenchmarkMode,
) -> tuple[list[DriftCase], list[BenchmarkCaseResult]]:
    """Run verification-drift evaluation for the selected repo pairs."""
    cases: list[DriftCase] = []
    results: list[BenchmarkCaseResult] = []
    for repo_spec, pair in selected_repo_pairs:
        repo_root = repo_manager.materialize_repo(repo_spec, output_dir / "corpus")
        repo_manager.git_checkout(repo_root, pair.base_ref)
        base_symbols = symbol_indexer.extract_symbols(repo_root)
        repo_manager.git_checkout(repo_root, pair.head_ref)
        head_symbols = symbol_indexer.extract_symbols(repo_root)
        pair_cases = symbol_indexer.build_drift_cases(
            repo_spec.repo_name,
            pair,
            base_symbols,
            head_symbols,
        )
        if mode == "no_memory":
            cases.extend(pair_cases)
            continue

        repo_manager.git_checkout(repo_root, pair.base_ref)
        kernel, metadata, artifacts, cache, scope = kernel_factory.open_kernel(
            repo_root=repo_root,
            repo_name=repo_spec.repo_name,
            pair=pair,
            output_dir=output_dir,
        )
        try:
            kernel_factory.seed_memories(
                kernel,
                repo_root,
                repo_spec.repo_name,
                pair,
                base_symbols,
                scope,
            )
            repo_manager.git_checkout(repo_root, pair.head_ref)
            dependency = kernel_factory.dependency_builder().build(
                str(repo_root),
                dependency_ids=(f"benchmark:{pair.pair_id}",),
                input_digests=(pair.head_ref,),
            )
            pair_results: list[BenchmarkCaseResult] = []
            for case in pair_cases:
                started = time.perf_counter()
                updated_memory, verification = kernel.verify_memory(case.memory_id, dependency)
                latency_ms = (time.perf_counter() - started) * 1000
                pair_results.append(
                    BenchmarkCaseResult(
                        suite="drift",
                        mode=mode,
                        case_id=case.case_id,
                        repo_name=case.repo_name,
                        commit_pair_id=case.commit_pair_id,
                        metrics={
                            "expected_status": case.expected_status.value,
                            "predicted_status": verification.current_status.value,
                            "latency_ms": latency_ms,
                        },
                        metadata={
                            "relative_path": case.relative_path,
                            "symbol": case.symbol,
                            "reason": updated_memory.validity.reason,
                        },
                    )
                )
            cases.extend(pair_cases)
            results.extend(pair_results)
        finally:
            kernel_factory.close_kernel_stores(metadata, artifacts, cache)
    return cases, results


__all__ = ["run_drift_suite"]
