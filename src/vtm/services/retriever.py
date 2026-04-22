"""Deterministic lexical retrieval helpers and retriever contracts."""

from __future__ import annotations

import re
from collections.abc import Mapping
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
SEARCHABLE_METADATA_KEYS = (
    "problem_id",
    "function_name",
    "feedback_signature",
    "memory_role",
)
EXACT_MATCH_BOOSTS = {
    "problem_id": 6.0,
    "function_name": 4.0,
    "feedback_signature": 3.0,
    "anchor_symbol": 3.0,
    "anchor_path": 2.0,
}
PARTIAL_MATCH_BOOSTS = {
    "feedback_signature": 0.75,
    "anchor_path": 0.5,
}


class Retriever(Protocol):
    """Minimal retrieval contract consumed by the kernel."""

    def retrieve(self, request: RetrieveRequest) -> RetrieveResult: ...

    def expand(self, memory_id: str) -> tuple[EvidenceRef, ...]: ...


def _tokenize(text: str) -> tuple[str, ...]:
    """Tokenize free text into case-folded alphanumeric terms."""
    return tuple(token.lower() for token in TOKEN_RE.findall(text))


def _searchable_fields(item: MemoryItem) -> dict[str, str]:
    """Collect the text fields that participate in lexical matching."""
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

    fields = {
        "title": item.title,
        "summary": item.summary,
        "tags": " ".join(item.tags),
        "payload": " ".join(payload_parts),
    }
    fields.update(_searchable_metadata_fields(item))
    fields.update(_searchable_anchor_fields(item))
    return fields


def _query_tokens(query: str) -> set[str]:
    """Return the unique lexical tokens present in a query string."""
    return set(_tokenize(query))


def _match_fields(
    query_tokens: set[str],
    item: MemoryItem,
) -> tuple[tuple[str, ...], tuple[str, ...], float]:
    """Score one memory item by token overlap across searchable fields."""
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


def _searchable_metadata_fields(item: MemoryItem) -> dict[str, str]:
    metadata = item.metadata if isinstance(item.metadata, Mapping) else {}
    fields: dict[str, str] = {}
    for key in SEARCHABLE_METADATA_KEYS:
        raw_value = metadata.get(key)
        if isinstance(raw_value, str | int | float):
            text = str(raw_value).strip()
            if text:
                fields[f"metadata:{key}"] = text
    return fields


def _searchable_anchor_fields(item: MemoryItem) -> dict[str, str]:
    path, symbol = _anchor_metadata(item)
    fields: dict[str, str] = {}
    if path:
        fields["anchor_path"] = path
    if symbol:
        fields["anchor_symbol"] = symbol
    return fields


def _anchor_metadata(item: MemoryItem) -> tuple[str | None, str | None]:
    for evidence in item.evidence:
        if evidence.code_anchor is not None:
            return evidence.code_anchor.path, evidence.code_anchor.symbol
    return None, None


def _value_tokens(value: Any) -> set[str]:
    if not isinstance(value, str | int | float):
        return set()
    return set(_tokenize(str(value)))


def _boosted_match_fields(
    query_tokens: set[str],
    item: MemoryItem,
) -> tuple[tuple[str, ...], tuple[str, ...], float, dict[str, float]]:
    metadata = item.metadata if isinstance(item.metadata, Mapping) else {}
    matched_fields: list[str] = []
    matched_tokens: set[str] = set()
    boost_breakdown: dict[str, float] = {}
    total_boost = 0.0

    for key in ("problem_id", "function_name", "feedback_signature"):
        value_tokens = _value_tokens(metadata.get(key))
        if not value_tokens:
            continue
        overlap = query_tokens & value_tokens
        if not overlap:
            continue
        field_name = f"metadata:{key}"
        matched_fields.append(field_name)
        matched_tokens.update(overlap)
        if value_tokens <= query_tokens:
            boost = EXACT_MATCH_BOOSTS[key]
        else:
            per_token = PARTIAL_MATCH_BOOSTS.get(key)
            boost = min(len(overlap) * per_token, EXACT_MATCH_BOOSTS[key]) if per_token else 0.0
        if boost > 0.0:
            total_boost += boost
            boost_breakdown[field_name] = round(boost, 6)

    anchor_path, anchor_symbol = _anchor_metadata(item)
    anchor_fields = {
        "anchor_symbol": _value_tokens(anchor_symbol),
        "anchor_path": _value_tokens(anchor_path),
    }
    for field_name, value_tokens in anchor_fields.items():
        if not value_tokens:
            continue
        overlap = query_tokens & value_tokens
        if not overlap:
            continue
        matched_fields.append(field_name)
        matched_tokens.update(overlap)
        if value_tokens <= query_tokens:
            boost = EXACT_MATCH_BOOSTS[field_name]
        else:
            per_token = PARTIAL_MATCH_BOOSTS.get(field_name)
            boost = (
                min(len(overlap) * per_token, EXACT_MATCH_BOOSTS[field_name])
                if per_token
                else 0.0
            )
        if boost > 0.0:
            total_boost += boost
            boost_breakdown[field_name] = round(boost, 6)

    return (
        tuple(sorted(matched_tokens)),
        tuple(sorted(set(matched_fields))),
        total_boost,
        boost_breakdown,
    )


def _candidate_evidence(
    request: RetrieveRequest,
    item: MemoryItem,
) -> tuple[tuple[EvidenceRef, ...], bool]:
    """Select evidence payloads according to the request evidence budget."""
    evidence: tuple[EvidenceRef, ...] = ()
    raw_evidence_available = False
    if request.evidence_budget is EvidenceBudget.FORCE_RAW:
        evidence = item.evidence
        raw_evidence_available = bool(item.evidence)
    elif request.evidence_budget is EvidenceBudget.SUMMARY_FIRST:
        raw_evidence_available = bool(item.evidence)
    return evidence, raw_evidence_available


def _explanation_metadata(item: MemoryItem) -> dict[str, Any]:
    """Expose procedure-specific metadata used by retrieval explanations."""
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
            matched_tokens, matched_fields, base_score = _match_fields(query_tokens, memory)
            boosted_tokens, boosted_fields, boost_score, boost_breakdown = _boosted_match_fields(
                query_tokens,
                memory,
            )
            score = base_score + boost_score
            all_matched_tokens = tuple(sorted(set(matched_tokens) | set(boosted_tokens)))
            all_matched_fields = tuple(dict.fromkeys((*matched_fields, *boosted_fields)))

            if query_tokens and score <= 0.0:
                continue

            evidence, raw_evidence_available = _candidate_evidence(request, memory)
            explanation_metadata = {
                **_explanation_metadata(memory),
                "lexical_base_score": base_score,
                "lexical_boost_score": boost_score,
            }
            if boost_breakdown:
                explanation_metadata["lexical_boost_fields"] = boost_breakdown

            candidates.append(
                RetrieveCandidate(
                    memory=memory,
                    score=score,
                    explanation=RetrieveExplanation(
                        matched_tokens=all_matched_tokens,
                        matched_fields=all_matched_fields,
                        score=score,
                        reason=(
                            "matched lexical overlap with exact-match boosts"
                            if boost_breakdown
                            else (
                                "matched lexical overlap"
                                if matched_tokens
                                else "returned deterministically for empty query"
                            )
                        ),
                        metadata=explanation_metadata,
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
