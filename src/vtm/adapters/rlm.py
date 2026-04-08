"""Provider-neutral reranking request and response models."""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import Field, model_validator

from vtm.base import VTMModel
from vtm.enums import ValidityStatus


class RLMRankedCandidate(VTMModel):
    """Candidate passed to or returned from an RLM reranker."""

    candidate_id: str
    title: str
    summary: str
    tags: tuple[str, ...] = Field(default_factory=tuple)
    status: ValidityStatus
    path: str | None = None
    symbol: str | None = None
    lexical_score: float = 0.0
    rlm_score: float | None = None
    final_score: float | None = None
    reason: str | None = None


class RLMRankRequest(VTMModel):
    """Reranking input for a lexical candidate set."""

    query: str
    candidates: tuple[RLMRankedCandidate, ...]
    top_k: int = Field(default=10, ge=1, le=100)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_candidate_ids(self) -> RLMRankRequest:
        """Require unique candidate identifiers in one request."""
        candidate_ids = [candidate.candidate_id for candidate in self.candidates]
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("RLM rank requests require unique candidate ids")
        return self


class RLMRankResponse(VTMModel):
    """Reranked candidate list returned by an adapter."""

    candidates: tuple[RLMRankedCandidate, ...] = Field(default_factory=tuple)
    model_name: str | None = None
    usage: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RLMAdapter(Protocol):
    """Interface implemented by provider-specific reranking adapters."""

    def rank_candidates(self, request: RLMRankRequest) -> RLMRankResponse: ...
