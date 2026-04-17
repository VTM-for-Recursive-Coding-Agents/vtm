"""Kernel-side retrieval helpers that persist retrieval side effects."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

from vtm.base import utc_now
from vtm.cache import CacheEntry, CacheKey
from vtm.enums import EvidenceBudget, ValidityStatus
from vtm.events import MemoryEvent
from vtm.evidence import EvidenceRef
from vtm.fingerprints import DependencyFingerprint
from vtm.memory_items import MemoryItem, MemoryStats
from vtm.retrieval import RetrieveRequest, RetrieveResult
from vtm.services.kernel_mutations import MetadataMutationRunner
from vtm.services.retriever import Retriever
from vtm.stores.base import CacheStore, EventStore, MetadataStore
from vtm.verification import VerificationResult


class RetrievalKernelOps:
    """Owns retrieval, evidence expansion, and retrieval-stat updates."""

    def __init__(
        self,
        *,
        metadata_store: MetadataStore,
        event_store: EventStore,
        cache_store: CacheStore,
        retriever: Retriever,
        mutations: MetadataMutationRunner,
        verify_memory: Callable[
            [str, DependencyFingerprint],
            tuple[MemoryItem, VerificationResult],
        ]
        | None = None,
    ) -> None:
        """Create retrieval helpers around stores, retriever, and cache."""
        self._metadata_store = metadata_store
        self._event_store = event_store
        self._cache_store = cache_store
        self._retriever = retriever
        self._mutations = mutations
        self._verify_memory = verify_memory

    def retrieve(self, request: RetrieveRequest) -> RetrieveResult:
        """Retrieve candidates and persist retrieval statistics."""
        result = self._retriever.retrieve(request)
        if request.verify_on_read:
            result = self._refresh_verified_candidates(result)
        retrieved_at = utc_now()
        updated_result, _events = self._mutations.run(
            lambda: self._persist_retrieval_result(result, request, retrieved_at),
            build_events=lambda persisted: persisted[1],
        )
        return updated_result

    def expand(self, memory_id: str) -> tuple[EvidenceRef, ...]:
        """Expand raw evidence for a memory item and emit an audit event."""
        evidence = self._retriever.expand(memory_id)
        if evidence:
            self._event_store.save_event(
                MemoryEvent(
                    event_type="memory_expanded",
                    memory_id=memory_id,
                    payload={"evidence_count": len(evidence)},
                )
            )
        return evidence

    def save_cache_entry(self, entry: CacheEntry) -> None:
        """Persist a cache entry through the configured cache store."""
        self._cache_store.save_cache_entry(entry)

    def get_cache_entry(self, key: CacheKey) -> CacheEntry | None:
        """Load a cache entry through the configured cache store."""
        return self._cache_store.get_cache_entry(key)

    def _increment_retrieval_stats(
        self,
        stats: MemoryStats,
        retrieved_at: datetime,
    ) -> MemoryStats:
        return stats.model_copy(
            update={
                "retrieval_count": stats.retrieval_count + 1,
                "last_retrieved_at": retrieved_at,
            }
        )

    def _persist_retrieval_result(
        self,
        result: RetrieveResult,
        request: RetrieveRequest,
        retrieved_at: datetime,
    ) -> tuple[RetrieveResult, tuple[MemoryEvent, ...]]:
        updated_candidates = []
        events: list[MemoryEvent] = []
        for candidate in result.candidates:
            stored = self._metadata_store.get_memory_item(candidate.memory.memory_id)
            if stored is None:
                updated_candidates.append(candidate)
                continue
            updated_memory = stored.model_copy(
                update={
                    "stats": self._increment_retrieval_stats(stored.stats, retrieved_at),
                    "updated_at": retrieved_at,
                }
            )
            self._metadata_store.save_memory_item(updated_memory)
            events.append(
                MemoryEvent(
                    event_type="memory_retrieved",
                    memory_id=updated_memory.memory_id,
                    payload={
                        "query": request.query,
                        "score": candidate.score,
                        "evidence_budget": request.evidence_budget.value,
                    },
                )
            )
            updated_candidates.append(candidate.model_copy(update={"memory": updated_memory}))
        return (
            result.model_copy(update={"candidates": tuple(updated_candidates)}),
            tuple(events),
        )

    def _refresh_verified_candidates(self, result: RetrieveResult) -> RetrieveResult:
        request = result.request
        current_dependency = request.current_dependency
        if current_dependency is None:
            return result
        if self._verify_memory is None:
            raise ValueError("verify_on_read retrieval requires a verification callback")

        updated_candidates = []
        verified_count = 0
        relocated_count = 0
        stale_filtered_count = 0
        for candidate in result.candidates:
            updated_memory, _verification = self._verify_memory(
                candidate.memory.memory_id,
                current_dependency,
            )
            evidence, raw_evidence_available = self._candidate_evidence_for_request(
                request,
                updated_memory,
            )
            updated_candidate = candidate.model_copy(
                update={
                    "memory": updated_memory,
                    "evidence": evidence,
                    "raw_evidence_available": raw_evidence_available,
                }
            )
            status = updated_memory.validity.status
            if request.return_verified_only and status not in {
                ValidityStatus.VERIFIED,
                ValidityStatus.RELOCATED,
            }:
                stale_filtered_count += 1
                continue
            if status is ValidityStatus.VERIFIED:
                verified_count += 1
            elif status is ValidityStatus.RELOCATED:
                relocated_count += 1
            updated_candidates.append(updated_candidate)

        inspected_count = len(result.candidates)
        stale_hit_rate = (
            float(stale_filtered_count) / float(inspected_count) if inspected_count else 0.0
        )
        return result.model_copy(
            update={
                "candidates": tuple(updated_candidates),
                "verified_count": verified_count,
                "relocated_count": relocated_count,
                "stale_filtered_count": stale_filtered_count,
                "stale_hit_rate": stale_hit_rate,
            }
        )

    def _candidate_evidence_for_request(
        self,
        request: RetrieveRequest,
        memory: MemoryItem,
    ) -> tuple[tuple[EvidenceRef, ...], bool]:
        if request.evidence_budget is EvidenceBudget.FORCE_RAW:
            return memory.evidence, bool(memory.evidence)
        if request.evidence_budget is EvidenceBudget.SUMMARY_FIRST:
            return (), bool(memory.evidence)
        return (), False
