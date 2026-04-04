from __future__ import annotations

from typing import Any

from pydantic import Field

from vtm.base import VTMModel
from vtm.enums import DetailLevel, EvidenceBudget, ValidityStatus
from vtm.evidence import EvidenceRef
from vtm.memory_items import MemoryItem, VisibilityScope


class RetrieveRequest(VTMModel):
    query: str
    scopes: tuple[VisibilityScope, ...]
    statuses: tuple[ValidityStatus, ...] | None = None
    detail_level: DetailLevel = DetailLevel.SUMMARY
    evidence_budget: EvidenceBudget = EvidenceBudget.SUMMARY_FIRST
    limit: int = Field(default=10, ge=1, le=100)
    allow_quarantined: bool = False


class RetrieveExplanation(VTMModel):
    matched_tokens: tuple[str, ...] = Field(default_factory=tuple)
    matched_fields: tuple[str, ...] = Field(default_factory=tuple)
    score: float
    reason: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrieveCandidate(VTMModel):
    memory: MemoryItem
    score: float
    explanation: RetrieveExplanation
    evidence: tuple[EvidenceRef, ...] = Field(default_factory=tuple)
    raw_evidence_available: bool = False


class RetrieveResult(VTMModel):
    request: RetrieveRequest
    candidates: tuple[RetrieveCandidate, ...] = Field(default_factory=tuple)
    total_candidates: int = Field(default=0, ge=0)
