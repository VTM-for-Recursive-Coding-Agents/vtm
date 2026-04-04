from __future__ import annotations

import hashlib
import json
import math
import shutil
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, TypeVar

from vtm.adapters.embeddings import DeterministicHashEmbeddingAdapter, EmbeddingAdapter
from vtm.adapters.git import GitRepoFingerprintCollector
from vtm.adapters.python_ast import PythonAstSyntaxAdapter
from vtm.adapters.rlm import RLMAdapter
from vtm.adapters.runtime import RuntimeEnvFingerprintCollector
from vtm.adapters.tree_sitter import PythonTreeSitterSyntaxAdapter
from vtm.benchmarks.executor import BenchmarkExecutor, SubprocessBenchmarkExecutor
from vtm.benchmarks.models import (
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
from vtm.benchmarks.swebench_harness import SWEbenchHarnessRunner
from vtm.benchmarks.symbol_index import SymbolIndexer, SymbolSnapshot
from vtm.enums import EvidenceBudget, EvidenceKind, MemoryKind, ScopeKind, ValidityStatus
from vtm.memory_items import ClaimPayload, MemoryItem, ValidityState, VisibilityScope
from vtm.retrieval import RetrieveRequest, RetrieveResult
from vtm.services import (
    BasicVerifier,
    DependencyFingerprintBuilder,
    EmbeddingRetriever,
    LexicalRetriever,
    RLMRerankingRetriever,
    TransactionalMemoryKernel,
)
from vtm.stores import (
    FilesystemArtifactStore,
    SqliteCacheStore,
    SqliteEmbeddingIndexStore,
    SqliteMetadataStore,
)

CaseT = TypeVar("CaseT", RetrievalCase, DriftCase)


class BenchmarkSuiteExecutor:
    def __init__(
        self,
        *,
        manifest: BenchmarkManifest,
        config: BenchmarkRunConfig,
        repo_manager: RepoWorkspaceManager,
        symbol_indexer: SymbolIndexer,
        rlm_adapter: RLMAdapter | None = None,
        embedding_adapter: EmbeddingAdapter | None = None,
    ) -> None:
        self._manifest = manifest
        self._config = config
        self._repo_manager = repo_manager
        self._symbol_indexer = symbol_indexer
        self._rlm_adapter = rlm_adapter
        self._embedding_adapter = embedding_adapter

    def run_retrieval_suite(
        self,
        output_dir: Path,
    ) -> tuple[list[RetrievalCase], list[BenchmarkCaseResult]]:
        cases: list[RetrievalCase] = []
        results: list[BenchmarkCaseResult] = []
        for repo_spec, pair in self._iter_selected_repo_pairs():
            repo_root = self._repo_manager.materialize_repo(repo_spec, output_dir / "corpus")
            pair_cases, pair_results = self._evaluate_retrieval_pair(
                repo_root,
                repo_spec,
                pair,
                output_dir,
            )
            cases.extend(pair_cases)
            results.extend(pair_results)
        return self._limit_cases(cases, results)

    def run_drift_suite(
        self,
        output_dir: Path,
    ) -> tuple[list[DriftCase], list[BenchmarkCaseResult]]:
        cases: list[DriftCase] = []
        results: list[BenchmarkCaseResult] = []
        for repo_spec, pair in self._iter_selected_repo_pairs():
            repo_root = self._repo_manager.materialize_repo(repo_spec, output_dir / "corpus")
            pair_cases, pair_results = self._evaluate_drift_pair(
                repo_root,
                repo_spec,
                pair,
                output_dir,
            )
            cases.extend(pair_cases)
            results.extend(pair_results)
        return self._limit_cases(cases, results)

    def run_coding_suite(
        self,
        output_dir: Path,
    ) -> tuple[list[CodingTaskCase], list[BenchmarkCaseResult]]:
        allowed_pairs = {
            (repo_spec.repo_name, pair.pair_id)
            for repo_spec, pair in self._iter_selected_repo_pairs()
        }
        cases = [
            task
            for task in self._manifest.coding_tasks
            if not allowed_pairs or (task.repo_name, task.commit_pair_id) in allowed_pairs
        ]
        if self._config.max_cases is not None:
            cases = cases[: self._config.max_cases]

        results: list[BenchmarkCaseResult] = []
        repo_map = {repo.repo_name: repo for repo in self._manifest.repos}
        for task in cases:
            repo_spec = repo_map[task.repo_name]
            pair = self._require_pair(repo_spec, task.commit_pair_id)
            repo_root = self._repo_manager.materialize_repo(repo_spec, output_dir / "corpus")
            results.append(self._evaluate_coding_task(repo_root, repo_spec, pair, task, output_dir))
        harness_cases = [task for task in cases if task.evaluation_backend == "swebench_harness"]
        if harness_cases:
            normalized_results, _ = self._swebench_harness_runner(output_dir).evaluate_predictions(
                cases=harness_cases,
                results=results,
                config=self._config,
                output_dir=output_dir,
            )
            results = self._merge_swebench_results(cases, results, normalized_results)
        return cases, results

    def _evaluate_retrieval_pair(
        self,
        repo_root: Path,
        repo_spec: RepoSpec,
        pair: CommitPair,
        output_dir: Path,
    ) -> tuple[list[RetrievalCase], list[BenchmarkCaseResult]]:
        self._repo_manager.git_checkout(repo_root, pair.base_ref)
        base_symbols = self._symbol_indexer.extract_symbols(repo_root)
        cases = self._symbol_indexer.build_retrieval_cases(repo_spec.repo_name, pair, base_symbols)
        if self._config.mode == "no_memory":
            return cases, [
                BenchmarkCaseResult(
                    suite="retrieval",
                    mode=self._config.mode,
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
                    },
                )
                for case in cases
            ]

        kernel, metadata, artifacts, cache, embedding_index, scope = self._open_kernel(
            repo_root=repo_root,
            repo_name=repo_spec.repo_name,
            pair=pair,
            output_dir=output_dir,
        )
        try:
            self._seed_memories(kernel, repo_root, repo_spec.repo_name, pair, base_symbols, scope)
            artifact_bytes_per_memory = self._artifact_bytes_per_memory(
                artifacts,
                len(base_symbols),
            )
            results: list[BenchmarkCaseResult] = []
            for case in cases:
                started = time.perf_counter()
                retrieve_result = kernel.retrieve(
                    RetrieveRequest(
                        query=case.query,
                        scopes=(scope,),
                        evidence_budget=EvidenceBudget.SUMMARY_ONLY,
                        limit=self._config.top_k,
                    )
                )
                latency_ms = (time.perf_counter() - started) * 1000
                rank = self._rank_for_expected(case.expected_memory_ids, retrieve_result)
                results.append(
                    BenchmarkCaseResult(
                        suite="retrieval",
                        mode=self._config.mode,
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
                        },
                        metadata={
                            "slice_name": case.slice_name,
                            "relative_path": case.relative_path,
                            "symbol": case.symbol,
                            "returned_memory_ids": [
                                candidate.memory.memory_id
                                for candidate in retrieve_result.candidates
                            ]
                        },
                    )
                )
            return cases, results
        finally:
            self._close_kernel_stores(metadata, artifacts, cache, embedding_index)

    def _evaluate_drift_pair(
        self,
        repo_root: Path,
        repo_spec: RepoSpec,
        pair: CommitPair,
        output_dir: Path,
    ) -> tuple[list[DriftCase], list[BenchmarkCaseResult]]:
        self._repo_manager.git_checkout(repo_root, pair.base_ref)
        base_symbols = self._symbol_indexer.extract_symbols(repo_root)
        self._repo_manager.git_checkout(repo_root, pair.head_ref)
        head_symbols = self._symbol_indexer.extract_symbols(repo_root)
        cases = self._symbol_indexer.build_drift_cases(
            repo_spec.repo_name,
            pair,
            base_symbols,
            head_symbols,
        )
        if self._config.mode == "no_memory":
            return cases, []

        self._repo_manager.git_checkout(repo_root, pair.base_ref)
        kernel, metadata, artifacts, cache, embedding_index, scope = self._open_kernel(
            repo_root=repo_root,
            repo_name=repo_spec.repo_name,
            pair=pair,
            output_dir=output_dir,
        )
        try:
            self._seed_memories(kernel, repo_root, repo_spec.repo_name, pair, base_symbols, scope)
            self._repo_manager.git_checkout(repo_root, pair.head_ref)
            dependency = self._dependency_builder().build(
                str(repo_root),
                dependency_ids=(f"benchmark:{pair.pair_id}",),
                input_digests=(pair.head_ref,),
            )
            results: list[BenchmarkCaseResult] = []
            for case in cases:
                started = time.perf_counter()
                updated_memory, verification = kernel.verify_memory(case.memory_id, dependency)
                latency_ms = (time.perf_counter() - started) * 1000
                results.append(
                    BenchmarkCaseResult(
                        suite="drift",
                        mode=self._config.mode,
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
            return cases, results
        finally:
            self._close_kernel_stores(metadata, artifacts, cache, embedding_index)

    def _evaluate_coding_task(
        self,
        repo_root: Path,
        repo_spec: RepoSpec,
        pair: CommitPair,
        task: CodingTaskCase,
        output_dir: Path,
    ) -> BenchmarkCaseResult:
        self._repo_manager.git_checkout(repo_root, pair.base_ref)
        memory_context: list[dict[str, Any]] = []
        expected_changed_paths = task.expected_changed_paths or task.touched_paths
        if self._config.mode != "no_memory":
            base_symbols = self._symbol_indexer.extract_symbols(repo_root)
            kernel, metadata, artifacts, cache, embedding_index, scope = self._open_kernel(
                repo_root=repo_root,
                repo_name=repo_spec.repo_name,
                pair=pair,
                output_dir=output_dir,
            )
            try:
                self._seed_memories(
                    kernel,
                    repo_root,
                    repo_spec.repo_name,
                    pair,
                    base_symbols,
                    scope,
                )
                query = " ".join(
                    [task.task_statement, *task.failing_tests, *task.touched_paths]
                ).strip()
                retrieval_result = kernel.retrieve(
                    RetrieveRequest(
                        query=query,
                        scopes=(scope,),
                        evidence_budget=EvidenceBudget.SUMMARY_ONLY,
                        limit=self._config.top_k,
                    )
                )
                memory_context = [
                    self._memory_context_payload(candidate)
                    for candidate in retrieval_result.candidates
                ]
            finally:
                self._close_kernel_stores(metadata, artifacts, cache, embedding_index)

        touched_paths = task.touched_paths or self._repo_manager.git_diff_paths(
            repo_root,
            pair.base_ref,
            pair.head_ref,
        )
        target_patch = task.target_patch or self._repo_manager.git_diff_text(
            repo_root,
            pair.base_ref,
            pair.head_ref,
        )
        target_patch_digest = hashlib.sha256(target_patch.encode("utf-8")).hexdigest()

        task_payload = {
            "case_id": task.case_id,
            "repo_name": task.repo_name,
            "commit_pair_id": task.commit_pair_id,
            "evaluation_backend": task.evaluation_backend,
            "instance_id": task.instance_id,
            "dataset_name": task.dataset_name,
            "base_ref": pair.base_ref,
            "head_ref": pair.head_ref,
            "commit_pair_label": pair.label,
            "task_statement": task.task_statement,
            "problem_statement": task.problem_statement,
            "hints_text": task.hints_text,
            "failing_tests": list(task.failing_tests),
            "fail_to_pass_tests": list(task.fail_to_pass_tests),
            "pass_to_pass_tests": list(task.pass_to_pass_tests),
            "expected_changed_paths": list(expected_changed_paths),
            "touched_paths": list(touched_paths),
            "test_command": list(task.test_command),
            "target_patch_digest": target_patch_digest,
            "gold_test_patch_digest": task.gold_test_patch_digest,
            "memory_mode": self._config.mode,
            "top_k": self._config.top_k,
            "task_kind": task.task_kind,
            "difficulty": task.difficulty,
            "memory_context": memory_context,
        }
        tasks_dir = output_dir / "task-packs"
        tasks_dir.mkdir(parents=True, exist_ok=True)
        task_file = tasks_dir / f"{task.case_id}.json"
        task_file.write_text(json.dumps(task_payload, indent=2, sort_keys=True), encoding="utf-8")

        executed = False
        passed = False
        patch_similarity: float | None = None
        runtime_ms = 0.0
        produced_patch_nonempty = False
        executor_succeeded = False
        evaluated = False
        resolved = False
        patch_applied = False
        changed_path_precision: float | None = None
        changed_path_recall: float | None = None
        changed_path_f1: float | None = None
        produced_changed_paths: tuple[str, ...] = ()
        produced_patch_text = ""
        testable = task.evaluation_backend == "swebench_harness" or bool(task.test_command)
        incomplete = not testable
        executor_metadata: dict[str, Any] = {}
        if self._config.executor_command:
            workspace_root = output_dir / "workspaces" / self._config.mode / task.case_id
            if workspace_root.exists():
                shutil.rmtree(workspace_root)
            self._repo_manager.run(
                [
                    "git",
                    "clone",
                    "--quiet",
                    str(repo_root),
                    str(workspace_root),
                ]
            )
            self._repo_manager.git_checkout(workspace_root, pair.base_ref)
            command = [
                token.format(task_file=str(task_file), workspace=str(workspace_root))
                for token in self._config.executor_command
            ]
            executor = self._executor(output_dir)
            outcome = executor.execute(
                case_id=task.case_id,
                command=tuple(command),
                workspace_root=workspace_root,
                task_file=task_file,
                test_command=task.test_command,
            )
            executed = True
            incomplete = (
                task.evaluation_backend == "local_subprocess"
                and not bool(task.test_command)
            )
            runtime_ms = outcome.runtime_ms
            executor_succeeded = outcome.command_exit_code == 0
            produced_patch_nonempty = bool(outcome.produced_patch_text.strip())
            if task.test_command:
                passed = outcome.test_exit_code == 0
                evaluated = True
                patch_applied = produced_patch_nonempty
            patch_similarity = SequenceMatcher(
                a=target_patch,
                b=outcome.produced_patch_text,
            ).ratio()
            produced_changed_paths = outcome.produced_changed_paths
            produced_patch_text = outcome.produced_patch_text
            (
                changed_path_precision,
                changed_path_recall,
                changed_path_f1,
            ) = self._changed_path_metrics(
                expected_changed_paths=expected_changed_paths,
                produced_changed_paths=produced_changed_paths,
            )
            executor_metadata = {
                "executor_command": list(outcome.command),
                "executor_exit_code": outcome.command_exit_code,
                "executor_runtime_ms": outcome.runtime_ms,
                "executor_stdout_path": outcome.command_stdout_path,
                "executor_stderr_path": outcome.command_stderr_path,
                "workspace": outcome.workspace,
                "task_file": outcome.task_file,
                "test_command": list(outcome.test_command),
                "test_exit_code": outcome.test_exit_code,
                "test_stdout_path": outcome.test_stdout_path,
                "test_stderr_path": outcome.test_stderr_path,
                "produced_patch_path": outcome.produced_patch_path,
                "produced_patch_digest": outcome.produced_patch_digest,
                "produced_changed_paths": list(outcome.produced_changed_paths),
                "produced_patch_text": outcome.produced_patch_text,
            }

        context_chars = len(json.dumps(memory_context, sort_keys=True))
        return BenchmarkCaseResult(
            suite="coding",
            mode=self._config.mode,
            case_id=task.case_id,
            repo_name=task.repo_name,
            commit_pair_id=task.commit_pair_id,
            metrics={
                "executed": executed,
                "evaluated": evaluated,
                "passed": passed,
                "resolved": resolved,
                "testable": testable,
                "incomplete": incomplete,
                "executor_succeeded": executor_succeeded,
                "produced_patch_nonempty": produced_patch_nonempty,
                "patch_applied": patch_applied,
                "changed_path_precision": changed_path_precision,
                "changed_path_recall": changed_path_recall,
                "changed_path_f1": changed_path_f1,
                "runtime_ms": runtime_ms,
                "patch_similarity": patch_similarity,
                "retrieval_usage_rate": 0.0 if not memory_context else 1.0,
                "context_chars": context_chars,
            },
            metadata={
                "task_file": str(task_file),
                "touched_paths": list(touched_paths),
                "expected_changed_paths": list(expected_changed_paths),
                "target_patch_digest": target_patch_digest,
                "memory_context_count": len(memory_context),
                "memory_mode": self._config.mode,
                "top_k": self._config.top_k,
                "task_kind": task.task_kind,
                "difficulty": task.difficulty,
                "evaluation_backend": task.evaluation_backend,
                "instance_id": task.instance_id,
                "dataset_name": task.dataset_name,
                "problem_statement": task.problem_statement,
                "hints_text": task.hints_text,
                "fail_to_pass_tests": list(task.fail_to_pass_tests),
                "pass_to_pass_tests": list(task.pass_to_pass_tests),
                "gold_test_patch_digest": task.gold_test_patch_digest,
                "base_ref": pair.base_ref,
                "head_ref": pair.head_ref,
                "commit_pair_label": pair.label,
                "produced_changed_paths": list(produced_changed_paths),
                "produced_patch_text": produced_patch_text,
                **executor_metadata,
            },
        )

    def _open_kernel(
        self,
        *,
        repo_root: Path,
        repo_name: str,
        pair: CommitPair,
        output_dir: Path,
    ) -> tuple[
        TransactionalMemoryKernel,
        SqliteMetadataStore,
        FilesystemArtifactStore,
        SqliteCacheStore,
        SqliteEmbeddingIndexStore | None,
        VisibilityScope,
    ]:
        store_root = (
            output_dir
            / ".vtm"
            / self._config.suite
            / self._config.mode
            / repo_name
            / pair.pair_id
        )
        if store_root.exists():
            shutil.rmtree(store_root)
        metadata = SqliteMetadataStore(
            store_root / "metadata.sqlite",
            event_log_path=store_root / "events.jsonl",
        )
        artifacts = FilesystemArtifactStore(store_root / "artifacts")
        cache = SqliteCacheStore(store_root / "cache.sqlite", event_store=metadata)
        embedding_index: SqliteEmbeddingIndexStore | None = None
        anchor_adapter = PythonTreeSitterSyntaxAdapter(fallback=PythonAstSyntaxAdapter())
        retriever: LexicalRetriever | RLMRerankingRetriever | EmbeddingRetriever = (
            LexicalRetriever(metadata)
        )
        if self._config.mode == "embedding":
            embedding_index = SqliteEmbeddingIndexStore(store_root / "embeddings.sqlite")
            retriever = EmbeddingRetriever(
                metadata,
                embedding_index,
                self._embedding_adapter or DeterministicHashEmbeddingAdapter(),
            )
        if self._config.mode == "lexical_rlm_rerank":
            if self._rlm_adapter is None:
                raise ValueError("lexical_rlm_rerank mode requires an RLM adapter")
            retriever = RLMRerankingRetriever(
                retriever,
                self._rlm_adapter,
                top_k_lexical=max(self._config.top_k * 2, self._config.top_k),
                top_k_final=self._config.top_k,
                cache_store=cache,
                cache_repo_root=str(repo_root),
            )
        kernel = TransactionalMemoryKernel(
            metadata_store=metadata,
            event_store=metadata,
            artifact_store=artifacts,
            cache_store=cache,
            verifier=BasicVerifier(relocator=anchor_adapter),
            retriever=retriever,
            anchor_adapter=anchor_adapter,
        )
        scope = VisibilityScope(kind=ScopeKind.REPO, scope_id=repo_name)
        return kernel, metadata, artifacts, cache, embedding_index, scope

    def _seed_memories(
        self,
        kernel: TransactionalMemoryKernel,
        repo_root: Path,
        repo_name: str,
        pair: CommitPair,
        symbols: dict[tuple[str, str], SymbolSnapshot],
        scope: VisibilityScope,
    ) -> None:
        dependency = self._dependency_builder().build(
            str(repo_root),
            dependency_ids=(f"benchmark:{pair.pair_id}",),
            input_digests=(pair.base_ref,),
        )
        tx = kernel.begin_transaction(
            scope,
            metadata={"repo_name": repo_name, "pair_id": pair.pair_id},
        )
        sorted_symbols = sorted(
            symbols.values(),
            key=lambda item: (item.relative_path, item.qualname),
        )
        for symbol in sorted_symbols:
            memory_id = self._symbol_indexer.memory_id(
                repo_name,
                pair.pair_id,
                symbol.relative_path,
                symbol.qualname,
            )
            source_path = repo_root / symbol.relative_path
            anchor = kernel.build_code_anchor(str(source_path), symbol.qualname)
            source_bytes = source_path.read_bytes()
            if anchor.start_byte is not None and anchor.end_byte is not None:
                snippet_bytes = source_bytes[anchor.start_byte : anchor.end_byte]
            else:
                snippet_bytes = symbol.snippet.encode("utf-8")
            artifact = kernel.capture_artifact(
                snippet_bytes,
                content_type="text/x-python",
                tool_name="benchmark-source-snapshot",
                metadata={
                    "repo_name": repo_name,
                    "pair_id": pair.pair_id,
                    "relative_path": symbol.relative_path,
                    "symbol": symbol.qualname,
                },
            )
            memory = MemoryItem(
                memory_id=memory_id,
                kind=MemoryKind.CLAIM,
                title=f"{symbol.qualname} in {symbol.relative_path}",
                summary=symbol.summary,
                payload=ClaimPayload(claim=symbol.summary),
                evidence=(
                    kernel.artifact_evidence(
                        artifact,
                        label="source-snippet",
                        summary="Captured benchmark source snippet",
                    ),
                    kernel.anchor_evidence(
                        anchor,
                        label="symbol-anchor",
                        summary="Captured benchmark code anchor",
                    ),
                ),
                tags=(repo_name, symbol.relative_path, symbol.kind),
                visibility=scope,
                validity=ValidityState(
                    status=ValidityStatus.VERIFIED,
                    dependency_fingerprint=dependency,
                ),
            )
            kernel.stage_memory_item(tx.tx_id, memory)
        kernel.commit_transaction(tx.tx_id)

    def _rank_for_expected(
        self,
        expected_memory_ids: tuple[str, ...],
        retrieve_result: RetrieveResult,
    ) -> int | None:
        for index, candidate in enumerate(retrieve_result.candidates, start=1):
            if candidate.memory.memory_id in expected_memory_ids:
                return index
        return None

    def _artifact_bytes_per_memory(
        self,
        artifact_store: FilesystemArtifactStore,
        memory_count: int,
    ) -> float:
        if memory_count == 0:
            return 0.0
        total_bytes = sum(record.size_bytes for record in artifact_store.list_artifact_records())
        return total_bytes / memory_count

    def _memory_context_payload(self, candidate: Any) -> dict[str, Any]:
        anchor = self._first_anchor(candidate.memory)
        return {
            "memory_id": candidate.memory.memory_id,
            "title": candidate.memory.title,
            "summary": candidate.memory.summary,
            "score": candidate.score,
            "status": candidate.memory.validity.status.value,
            "relative_path": anchor.path if anchor is not None else None,
            "symbol": anchor.symbol if anchor is not None else None,
            "slice_name": candidate.explanation.metadata.get("slice_name"),
            "raw_anchor_path": anchor.path if anchor is not None else None,
        }

    def _first_anchor(self, memory: MemoryItem) -> Any | None:
        for evidence in memory.evidence:
            if evidence.kind is EvidenceKind.CODE_ANCHOR and evidence.code_anchor is not None:
                return evidence.code_anchor
        return None

    def _changed_path_metrics(
        self,
        *,
        expected_changed_paths: tuple[str, ...],
        produced_changed_paths: tuple[str, ...],
    ) -> tuple[float, float, float]:
        expected = set(expected_changed_paths)
        produced = set(produced_changed_paths)
        if not expected and not produced:
            return 1.0, 1.0, 1.0
        if not produced:
            return 0.0, 0.0, 0.0
        true_positives = len(expected & produced)
        precision = true_positives / len(produced)
        recall = 0.0 if not expected else true_positives / len(expected)
        if precision + recall == 0.0:
            f1 = 0.0
        else:
            f1 = 2.0 * precision * recall / (precision + recall)
        return precision, recall, f1

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

    def _dependency_builder(self) -> DependencyFingerprintBuilder:
        return DependencyFingerprintBuilder(
            repo_collector=GitRepoFingerprintCollector(),
            env_collector=RuntimeEnvFingerprintCollector(),
        )

    def _require_pair(self, repo_spec: RepoSpec, pair_id: str) -> CommitPair:
        for pair in repo_spec.commit_pairs:
            if pair.pair_id == pair_id:
                return pair
        raise KeyError(f"unknown commit pair: {repo_spec.repo_name}:{pair_id}")

    def _close_kernel_stores(
        self,
        metadata: SqliteMetadataStore,
        artifacts: FilesystemArtifactStore,
        cache: SqliteCacheStore,
        embedding_index: SqliteEmbeddingIndexStore | None,
    ) -> None:
        cache.close()
        if embedding_index is not None:
            embedding_index.close()
        artifacts.close()
        metadata.close()

    def _executor(self, output_dir: Path) -> BenchmarkExecutor:
        return SubprocessBenchmarkExecutor(
            repo_manager=self._repo_manager,
            output_root=output_dir,
        )

    def _swebench_harness_runner(self, output_dir: Path) -> SWEbenchHarnessRunner:
        return SWEbenchHarnessRunner()

    def _merge_swebench_results(
        self,
        cases: list[CodingTaskCase],
        results: list[BenchmarkCaseResult],
        normalized_results: dict[str, Any],
    ) -> list[BenchmarkCaseResult]:
        case_map = {case.case_id: case for case in cases}
        merged: list[BenchmarkCaseResult] = []
        for result in results:
            case = case_map[result.case_id]
            if case.evaluation_backend != "swebench_harness":
                merged.append(result)
                continue
            harness = normalized_results[case.instance_id or case.case_id]
            merged.append(
                result.model_copy(
                    update={
                        "metrics": {
                            **result.metrics,
                            "evaluated": True,
                            "passed": harness.resolved,
                            "resolved": harness.resolved,
                            "incomplete": False,
                            "patch_applied": harness.patch_applied,
                        },
                        "metadata": {
                            **result.metadata,
                            "harness_status": harness.harness_status,
                            "evaluation_log_path": harness.evaluation_log_path,
                        },
                    }
                )
            )
        return merged
