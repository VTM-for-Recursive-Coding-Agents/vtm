from __future__ import annotations

import hashlib
import math

from vtm.adapters.embeddings import EmbeddingAdapter
from vtm.base import utc_now
from vtm.embeddings import EmbeddingIndexEntry
from vtm.enums import EvidenceKind
from vtm.evidence import EvidenceRef
from vtm.memory_items import MemoryItem
from vtm.policies import DEFAULT_RETRIEVAL_STATUSES
from vtm.retrieval import RetrieveCandidate, RetrieveExplanation, RetrieveRequest, RetrieveResult
from vtm.services.retriever import (
    _candidate_evidence,
    _explanation_metadata,
    _match_fields,
    _query_tokens,
)
from vtm.stores.base import EmbeddingIndexStore, MetadataStore


class EmbeddingRetriever:
    def __init__(
        self,
        metadata_store: MetadataStore,
        index_store: EmbeddingIndexStore,
        embedding_adapter: EmbeddingAdapter,
    ) -> None:
        self._metadata_store = metadata_store
        self._index_store = index_store
        self._embedding_adapter = embedding_adapter

    @property
    def adapter_id(self) -> str:
        return self._embedding_adapter.adapter_id

    def retrieve(self, request: RetrieveRequest) -> RetrieveResult:
        statuses = request.statuses or DEFAULT_RETRIEVAL_STATUSES
        memories = self._metadata_store.query_memory_items(
            request.scopes,
            statuses=statuses,
            allow_quarantined=request.allow_quarantined,
        )
        query_tokens = _query_tokens(request.query)
        query_vector = self._embedding_adapter.embed_text(request.query)
        candidates: list[RetrieveCandidate] = []

        for memory in memories:
            matched_tokens, matched_fields, lexical_score = _match_fields(query_tokens, memory)
            indexed_entry = self._get_or_build_entry(memory)
            embedding_score = self._cosine_similarity(query_vector, indexed_entry.vector)
            if query_tokens and embedding_score <= 0.0 and lexical_score <= 0.0:
                continue

            evidence, raw_evidence_available = _candidate_evidence(request, memory)
            candidates.append(
                RetrieveCandidate(
                    memory=memory,
                    score=embedding_score,
                    explanation=RetrieveExplanation(
                        matched_tokens=matched_tokens,
                        matched_fields=matched_fields,
                        score=embedding_score,
                        reason=(
                            "matched embedding similarity"
                            if request.query.strip()
                            else "returned deterministically for empty query"
                        ),
                        metadata={
                            **_explanation_metadata(memory),
                            "embedding_adapter_id": self._embedding_adapter.adapter_id,
                            "embedding_content_digest": indexed_entry.content_digest,
                            "embedding_score": embedding_score,
                            "lexical_score": lexical_score,
                        },
                    ),
                    evidence=evidence,
                    raw_evidence_available=raw_evidence_available,
                )
            )

        candidates.sort(
            key=lambda candidate: (
                -candidate.score,
                -float(candidate.explanation.metadata.get("lexical_score", 0.0)),
                -candidate.memory.updated_at.timestamp(),
                candidate.memory.memory_id,
            )
        )
        limited = tuple(candidates[: request.limit])
        return RetrieveResult(request=request, candidates=limited, total_candidates=len(candidates))

    def expand(self, memory_id: str) -> tuple[EvidenceRef, ...]:
        memory = self._metadata_store.get_memory_item(memory_id)
        if memory is None:
            return ()
        return memory.evidence

    def _get_or_build_entry(self, memory: MemoryItem) -> EmbeddingIndexEntry:
        content = self._embedding_text(memory)
        content_digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        existing = self._index_store.get_entry(memory.memory_id, self._embedding_adapter.adapter_id)
        if existing is not None and existing.content_digest == content_digest:
            return existing

        now = utc_now()
        entry = EmbeddingIndexEntry(
            memory_id=memory.memory_id,
            adapter_id=self._embedding_adapter.adapter_id,
            content_digest=content_digest,
            vector=self._embedding_adapter.embed_text(content),
            created_at=existing.created_at if existing is not None else now,
            updated_at=now,
        )
        self._index_store.save_entry(entry)
        return entry

    def _embedding_text(self, memory: MemoryItem) -> str:
        code_anchor = next(
            (
                evidence.code_anchor
                for evidence in memory.evidence
                if evidence.kind is EvidenceKind.CODE_ANCHOR and evidence.code_anchor is not None
            ),
            None,
        )
        parts = [
            f"title: {memory.title}",
            f"summary: {memory.summary}",
            f"tags: {' '.join(memory.tags)}",
            f"status: {memory.validity.status.value}",
        ]
        if code_anchor is not None:
            parts.extend(
                [
                    f"path: {code_anchor.path}",
                    f"symbol: {code_anchor.symbol or ''}",
                    f"kind: {code_anchor.kind or ''}",
                    f"language: {code_anchor.language or ''}",
                ]
            )
        return "\n".join(parts)

    def _cosine_similarity(
        self,
        left: tuple[float, ...],
        right: tuple[float, ...],
    ) -> float:
        if not left or not right:
            return 0.0
        limit = min(len(left), len(right))
        if limit == 0:
            return 0.0
        left_norm = math.sqrt(sum(value * value for value in left[:limit]))
        right_norm = math.sqrt(sum(value * value for value in right[:limit]))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        dot = sum(left[index] * right[index] for index in range(limit))
        return dot / (left_norm * right_norm)
