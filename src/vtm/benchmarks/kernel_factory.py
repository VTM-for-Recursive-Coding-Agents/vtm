"""Helpers for constructing and seeding benchmark-local kernels."""

from __future__ import annotations

import shutil
from pathlib import Path

from vtm.adapters.embeddings import DeterministicHashEmbeddingAdapter, EmbeddingAdapter
from vtm.adapters.git import GitRepoFingerprintCollector
from vtm.adapters.python_ast import PythonAstSyntaxAdapter
from vtm.adapters.rlm import RLMAdapter
from vtm.adapters.runtime import RuntimeEnvFingerprintCollector
from vtm.adapters.tree_sitter import PythonTreeSitterSyntaxAdapter
from vtm.benchmarks.models import BenchmarkRunConfig, CommitPair
from vtm.benchmarks.symbol_index import SymbolIndexer, SymbolSnapshot
from vtm.enums import MemoryKind, ScopeKind, ValidityStatus
from vtm.memory_items import ClaimPayload, MemoryItem, ValidityState, VisibilityScope
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


class BenchmarkKernelFactory:
    """Creates isolated kernels and seeds benchmark source memory into them."""

    def __init__(
        self,
        *,
        config: BenchmarkRunConfig,
        symbol_indexer: SymbolIndexer,
        rlm_adapter: RLMAdapter | None = None,
        embedding_adapter: EmbeddingAdapter | None = None,
    ) -> None:
        """Bind run config plus optional retrieval adapters."""
        self._config = config
        self._symbol_indexer = symbol_indexer
        self._rlm_adapter = rlm_adapter
        self._embedding_adapter = embedding_adapter

    def open_kernel(
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
        """Create a fresh benchmark-local kernel store topology."""
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

    def seed_memories(
        self,
        kernel: TransactionalMemoryKernel,
        repo_root: Path,
        repo_name: str,
        pair: CommitPair,
        symbols: dict[tuple[str, str], SymbolSnapshot],
        scope: VisibilityScope,
    ) -> None:
        """Seed symbol snapshots from the base repo state into the kernel."""
        dependency = self.dependency_builder().build(
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

    def artifact_bytes_per_memory(
        self,
        artifact_store: FilesystemArtifactStore,
        memory_count: int,
    ) -> float:
        """Compute average captured artifact bytes per seeded memory."""
        if memory_count == 0:
            return 0.0
        total_bytes = sum(record.size_bytes for record in artifact_store.list_artifact_records())
        return total_bytes / memory_count

    def dependency_builder(self) -> DependencyFingerprintBuilder:
        """Create the dependency builder used by benchmark kernels."""
        return DependencyFingerprintBuilder(
            repo_collector=GitRepoFingerprintCollector(),
            env_collector=RuntimeEnvFingerprintCollector(),
        )

    def close_kernel_stores(
        self,
        metadata: SqliteMetadataStore,
        artifacts: FilesystemArtifactStore,
        cache: SqliteCacheStore,
        embedding_index: SqliteEmbeddingIndexStore | None,
    ) -> None:
        """Close all stores opened for a benchmark-local kernel."""
        cache.close()
        if embedding_index is not None:
            embedding_index.close()
        artifacts.close()
        metadata.close()


__all__ = ["BenchmarkKernelFactory"]
