"""Deterministic lexical retrieval helpers and retriever contracts."""

from __future__ import annotations

import re
from typing import Any, Protocol

from vtm.enums import EvidenceBudget
from vtm.evidence import EvidenceRef
from vtm.memory_items import (
    ClaimPayload,
    ConstraintPayload,
    DecisionPayload,
    MemoryItem,
    ProcedurePayload,
)
from vtm.policies import DEFAULT_RETRIEVAL_STATUSES
from vtm.retrieval import (
    RetrieveCandidate,
    RetrieveExplanation,
    RetrieveRequest,
    RetrieveResult,
)
from vtm.stores.base import MetadataStore

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


class Retriever(Protocol):
    """Minimal retrieval contract consumed by the kernel."""

    def retrieve(self, request: RetrieveRequest) -> RetrieveResult: ...

    def expand(self, memory_id: str) -> tuple[EvidenceRef, ...]: ...


def _tokenize(text: str) -> tuple[str, ...]:
    return tuple(token.lower() for token in TOKEN_RE.findall(text))


def _searchable_fields(item: MemoryItem) -> dict[str, str]:
    payload_parts: list[str] = []
    if isinstance(item.payload, ClaimPayload):
        payload_parts.append(item.payload.claim)
    if isinstance(item.payload, ConstraintPayload):
        payload_parts.append(item.payload.statement)
    if isinstance(item.payload, DecisionPayload) and item.payload.rationale is not None:
        payload_parts.append(item.payload.rationale)
    if isinstance(item.payload, ProcedurePayload):
        payload_parts.append(item.payload.goal)
        payload_parts.extend(step.instruction for step in item.payload.steps)

    return {
        "title": item.title,
        "summary": item.summary,
        "tags": " ".join(item.tags),
        "payload": " ".join(payload_parts),
    }


def _query_tokens(query: str) -> set[str]:
    return set(_tokenize(query))


def _match_fields(
    query_tokens: set[str],
    item: MemoryItem,
) -> tuple[tuple[str, ...], tuple[str, ...], float]:
    fields = _searchable_fields(item)
    matched_tokens: set[str] = set()
    matched_fields: list[str] = []
    score = 0.0
    for field_name, text in fields.items():
        field_tokens = set(_tokenize(text))
        overlap = query_tokens & field_tokens if query_tokens else field_tokens
        if overlap:
            matched_tokens.update(overlap)
            matched_fields.append(field_name)
            score += float(len(overlap))
    return tuple(sorted(matched_tokens)), tuple(matched_fields), score


def _candidate_evidence(
    request: RetrieveRequest,
    item: MemoryItem,
) -> tuple[tuple[EvidenceRef, ...], bool]:
    evidence: tuple[EvidenceRef, ...] = ()
    raw_evidence_available = False
    if request.evidence_budget is EvidenceBudget.FORCE_RAW:
        evidence = item.evidence
        raw_evidence_available = bool(item.evidence)
    elif request.evidence_budget is EvidenceBudget.SUMMARY_FIRST:
        raw_evidence_available = bool(item.evidence)
    return evidence, raw_evidence_available


def _explanation_metadata(item: MemoryItem) -> dict[str, Any]:
    if not isinstance(item.payload, ProcedurePayload):
        return {}

    latest_validation = item.metadata.get("latest_procedure_validation")
    latest_status = item.validity.status.value
    latest_success: bool | None = None
    validator_kind = None
    if item.payload.validator is not None:
        validator_kind = item.payload.validator.kind
    if isinstance(latest_validation, dict):
        latest_status = str(latest_validation.get("status", latest_status))
        raw_success = latest_validation.get("success")
        latest_success = raw_success if isinstance(raw_success, bool) else None

    return {
        "procedure_has_validator": item.payload.validator is not None,
        "procedure_validator_kind": validator_kind,
        "latest_validation_status": latest_status,
        "latest_validation_success": latest_success,
    }


class LexicalRetriever:
    """Token-overlap retriever over committed visible memory."""

    def __init__(self, metadata_store: MetadataStore) -> None:
        """Bind the retriever to a metadata store."""
        self._metadata_store = metadata_store

    def retrieve(self, request: RetrieveRequest) -> RetrieveResult:
        """Return deterministically ranked candidates for a retrieval request."""
        statuses = request.statuses or DEFAULT_RETRIEVAL_STATUSES
        memories = self._metadata_store.query_memory_items(
            request.scopes,
            statuses=statuses,
            allow_quarantined=request.allow_quarantined,
        )
        query_tokens = _query_tokens(request.query)
        candidates: list[RetrieveCandidate] = []

        for memory in memories:
            matched_tokens, matched_fields, score = _match_fields(query_tokens, memory)

            if query_tokens and score <= 0.0:
                continue

            evidence, raw_evidence_available = _candidate_evidence(request, memory)

            candidates.append(
                RetrieveCandidate(
                    memory=memory,
                    score=score,
                    explanation=RetrieveExplanation(
                        matched_tokens=tuple(sorted(matched_tokens)),
                        matched_fields=tuple(matched_fields),
                        score=score,
                        reason=(
                            "matched lexical overlap"
                            if matched_tokens
                            else "returned deterministically for empty query"
                        ),
                        metadata=_explanation_metadata(memory),
                    ),
                    evidence=evidence,
                    raw_evidence_available=raw_evidence_available,
                )
            )

        candidates.sort(key=lambda candidate: (-candidate.score, candidate.memory.memory_id))
        limited = tuple(candidates[: request.limit])
        return RetrieveResult(request=request, candidates=limited, total_candidates=len(candidates))

    def expand(self, memory_id: str) -> tuple[EvidenceRef, ...]:
        """Return all raw evidence attached to a stored memory item."""
        memory = self._metadata_store.get_memory_item(memory_id)
        if memory is None:
            return ()
        return memory.evidence
