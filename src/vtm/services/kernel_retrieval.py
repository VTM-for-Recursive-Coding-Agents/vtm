from __future__ import annotations

from datetime import datetime

from vtm.base import utc_now
from vtm.cache import CacheEntry, CacheKey
from vtm.events import MemoryEvent
from vtm.evidence import EvidenceRef
from vtm.memory_items import MemoryStats
from vtm.retrieval import RetrieveRequest, RetrieveResult
from vtm.services.kernel_mutations import MetadataMutationRunner
from vtm.services.retriever import Retriever
from vtm.stores.base import CacheStore, EventStore, MetadataStore


class RetrievalKernelOps:
    def __init__(
        self,
        *,
        metadata_store: MetadataStore,
        event_store: EventStore,
        cache_store: CacheStore,
        retriever: Retriever,
        mutations: MetadataMutationRunner,
    ) -> None:
        self._metadata_store = metadata_store
        self._event_store = event_store
        self._cache_store = cache_store
        self._retriever = retriever
        self._mutations = mutations

    def retrieve(self, request: RetrieveRequest) -> RetrieveResult:
        result = self._retriever.retrieve(request)
        retrieved_at = utc_now()
        updated_result, _events = self._mutations.run(
            lambda: self._persist_retrieval_result(result, request, retrieved_at),
            build_events=lambda persisted: persisted[1],
        )
        return updated_result

    def expand(self, memory_id: str) -> tuple[EvidenceRef, ...]:
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
        self._cache_store.save_cache_entry(entry)

    def get_cache_entry(self, key: CacheKey) -> CacheEntry | None:
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
