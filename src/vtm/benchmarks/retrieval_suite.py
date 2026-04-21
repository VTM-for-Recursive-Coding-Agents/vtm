"""Retrieval benchmark suite execution helpers."""

from __future__ import annotations

import math
import time
from pathlib import Path

from vtm.benchmarks.kernel_factory import BenchmarkKernelFactory
from vtm.benchmarks.models import (
    BenchmarkCaseResult,
    BenchmarkMode,
    CommitPair,
    RepoSpec,
    RetrievalCase,
    resolved_benchmark_mode,
)
from vtm.benchmarks.repo_materialization import RepoWorkspaceManager
from vtm.benchmarks.symbol_index import SymbolIndexer
from vtm.enums import EvidenceBudget, ValidityStatus
from vtm.retrieval import RetrieveRequest, RetrieveResult


def run_retrieval_suite(
    *,
    output_dir: Path,
    selected_repo_pairs: list[tuple[RepoSpec, CommitPair]],
    repo_manager: RepoWorkspaceManager,
    symbol_indexer: SymbolIndexer,
    kernel_factory: BenchmarkKernelFactory,
    top_k: int,
    mode: BenchmarkMode,
    seed_on_base_query_on_head: bool = False,
) -> tuple[list[RetrievalCase], list[BenchmarkCaseResult]]:
    """Run retrieval evaluation for the selected repo and commit pairs."""
    resolved_mode = resolved_benchmark_mode(mode)
    cases: list[RetrievalCase] = []
    results: list[BenchmarkCaseResult] = []
    for repo_spec, pair in selected_repo_pairs:
        repo_root = repo_manager.materialize_repo(repo_spec, output_dir / "corpus")
        repo_manager.git_checkout(repo_root, pair.base_ref)
        base_symbols = symbol_indexer.extract_symbols(repo_root)
        head_symbols = None
        if seed_on_base_query_on_head:
            repo_manager.git_checkout(repo_root, pair.head_ref)
            head_symbols = symbol_indexer.extract_symbols(repo_root)
            repo_manager.git_checkout(repo_root, pair.base_ref)
        pair_cases = symbol_indexer.build_retrieval_cases(
            repo_spec.repo_name,
            pair,
            base_symbols,
            head_symbols=head_symbols,
        )
        query_ref = pair.head_ref if seed_on_base_query_on_head else pair.base_ref
        if resolved_mode == "no_memory":
            no_memory_results = [
                BenchmarkCaseResult(
                    suite="retrieval",
                    mode=mode,
                    case_id=case.case_id,
                    repo_name=case.repo_name,
                    commit_pair_id=case.commit_pair_id,
                    metrics={
                        "rank": None,
                        "recall_at_1": 0.0,
                        "recall_at_3": 0.0,
                        "recall_at_5": 0.0,
                        "mrr": 0.0,
                        "ndcg": 0.0,
                        "latency_ms": 0.0,
                        "artifact_bytes_per_memory": 0.0,
                        "verified_count": 0,
                        "relocated_count": 0,
                        "stale_filtered_count": 0,
                        "stale_hit_rate": 0.0,
                    },
                    metadata=_retrieval_metadata(
                        case,
                        seed_ref=pair.base_ref,
                        query_ref=query_ref,
                        seed_on_base_query_on_head=seed_on_base_query_on_head,
                        returned_memory_ids=[],
                    ),
                )
                for case in pair_cases
            ]
            cases.extend(pair_cases)
            results.extend(no_memory_results)
            continue

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
            artifact_bytes_per_memory = kernel_factory.artifact_bytes_per_memory(
                artifacts,
                len(base_symbols),
            )
            if seed_on_base_query_on_head:
                repo_manager.git_checkout(repo_root, pair.head_ref)
            pair_results: list[BenchmarkCaseResult] = []
            current_dependency = kernel_factory.dependency_builder().build(
                str(repo_root),
                dependency_ids=(f"benchmark:{pair.pair_id}",),
                input_digests=(query_ref,),
            )
            for case in pair_cases:
                started = time.perf_counter()
                retrieve_result = kernel.retrieve(
                    RetrieveRequest(
                        query=case.query,
                        scopes=(scope,),
                        statuses=(
                            tuple(ValidityStatus)
                            if resolved_mode == "verified_lexical"
                            else None
                        ),
                        evidence_budget=EvidenceBudget.SUMMARY_ONLY,
                        limit=top_k,
                        current_dependency=(
                            current_dependency
                            if resolved_mode == "verified_lexical"
                            else None
                        ),
                        verify_on_read=resolved_mode == "verified_lexical",
                        return_verified_only=resolved_mode == "verified_lexical",
                    )
                )
                latency_ms = (time.perf_counter() - started) * 1000
                rank = rank_for_expected(case.expected_memory_ids, retrieve_result)
                pair_results.append(
                    BenchmarkCaseResult(
                        suite="retrieval",
                        mode=mode,
                        case_id=case.case_id,
                        repo_name=case.repo_name,
                        commit_pair_id=case.commit_pair_id,
                        metrics={
                            "rank": rank,
                            "recall_at_1": 1.0 if rank is not None and rank <= 1 else 0.0,
                            "recall_at_3": 1.0 if rank is not None and rank <= 3 else 0.0,
                            "recall_at_5": 1.0 if rank is not None and rank <= 5 else 0.0,
                            "mrr": 0.0 if rank is None else 1.0 / rank,
                            "ndcg": 0.0 if rank is None else 1.0 / math.log2(rank + 1),
                            "latency_ms": latency_ms,
                            "artifact_bytes_per_memory": artifact_bytes_per_memory,
                            "verified_count": retrieve_result.verified_count,
                            "relocated_count": retrieve_result.relocated_count,
                            "stale_filtered_count": retrieve_result.stale_filtered_count,
                            "stale_hit_rate": retrieve_result.stale_hit_rate,
                        },
                        metadata=_retrieval_metadata(
                            case,
                            seed_ref=pair.base_ref,
                            query_ref=query_ref,
                            seed_on_base_query_on_head=seed_on_base_query_on_head,
                            returned_memory_ids=[
                                candidate.memory.memory_id
                                for candidate in retrieve_result.candidates
                            ],
                        ),
                    )
                )
            cases.extend(pair_cases)
            results.extend(pair_results)
        finally:
            kernel_factory.close_kernel_stores(metadata, artifacts, cache)
    return cases, results


def rank_for_expected(
    expected_memory_ids: tuple[str, ...],
    retrieve_result: RetrieveResult,
) -> int | None:
    """Return the first rank at which any expected memory appears."""
    for index, candidate in enumerate(retrieve_result.candidates, start=1):
        if candidate.memory.memory_id in expected_memory_ids:
            return index
    return None


def _retrieval_metadata(
    case: RetrievalCase,
    *,
    seed_ref: str,
    query_ref: str,
    seed_on_base_query_on_head: bool,
    returned_memory_ids: list[str],
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "slice_name": case.slice_name,
        "relative_path": case.relative_path,
        "symbol": case.symbol,
        "seed_ref": seed_ref,
        "query_ref": query_ref,
        "seed_on_base_query_on_head": seed_on_base_query_on_head,
        "returned_memory_ids": returned_memory_ids,
    }
    if case.expected_head_status is not None:
        metadata["expected_head_status"] = case.expected_head_status.value
    return metadata


__all__ = ["run_retrieval_suite"]
