"""Request and result models for retrieval operations."""

from __future__ import annotations

from typing import Any

from pydantic import Field, model_validator

from vtm.base import VTMModel
from vtm.enums import DetailLevel, EvidenceBudget, ValidityStatus
from vtm.evidence import EvidenceRef
from vtm.fingerprints import DependencyFingerprint
from vtm.memory_items import MemoryItem, VisibilityScope


class RetrieveRequest(VTMModel):
    """Inputs for a retrieval call against visible committed memory."""

    query: str
    scopes: tuple[VisibilityScope, ...]
    statuses: tuple[ValidityStatus, ...] | None = None
    detail_level: DetailLevel = DetailLevel.SUMMARY
    evidence_budget: EvidenceBudget = EvidenceBudget.SUMMARY_FIRST
    limit: int = Field(default=10, ge=1, le=100)
    allow_quarantined: bool = False
    current_dependency: DependencyFingerprint | None = None
    verify_on_read: bool = False
    return_verified_only: bool = False

    @model_validator(mode="after")
    def validate_verification_controls(self) -> RetrieveRequest:
        """Require a dependency fingerprint when retrieval verifies candidates."""
        if self.verify_on_read and self.current_dependency is None:
            raise ValueError("verify_on_read retrieval requires current_dependency")
        if self.return_verified_only and not self.verify_on_read:
            raise ValueError("return_verified_only retrieval requires verify_on_read")
        return self


class RetrieveExplanation(VTMModel):
    """Why a candidate was returned and how it was scored."""

    matched_tokens: tuple[str, ...] = Field(default_factory=tuple)
    matched_fields: tuple[str, ...] = Field(default_factory=tuple)
    score: float
    reason: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrieveCandidate(VTMModel):
    """One retrieved memory item plus score and attached evidence."""

    memory: MemoryItem
    score: float
    explanation: RetrieveExplanation
    evidence: tuple[EvidenceRef, ...] = Field(default_factory=tuple)
    raw_evidence_available: bool = False


class RetrieveResult(VTMModel):
    """Ordered retrieval response for a single query."""

    request: RetrieveRequest
    candidates: tuple[RetrieveCandidate, ...] = Field(default_factory=tuple)
    total_candidates: int = Field(default=0, ge=0)
    verified_count: int = Field(default=0, ge=0)
    relocated_count: int = Field(default=0, ge=0)
    stale_filtered_count: int = Field(default=0, ge=0)
    stale_hit_rate: float = Field(default=0.0, ge=0.0, le=1.0)
