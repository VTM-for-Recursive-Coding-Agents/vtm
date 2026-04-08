"""Durable memory-item payloads and supporting state records."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import Field, model_validator

from vtm.base import VTMModel, utc_now
from vtm.enums import (
    ClaimStrength,
    DetailLevel,
    EvidenceKind,
    MemoryKind,
    ScopeKind,
    ValidityStatus,
)
from vtm.evidence import EvidenceRef
from vtm.fingerprints import DependencyFingerprint
from vtm.ids import new_memory_id


class ProcedureStep(VTMModel):
    """One ordered instruction inside a procedure memory."""

    order: int = Field(ge=0)
    instruction: str
    expected_outcome: str | None = None


class ValidatorSpec(VTMModel):
    """Validator configuration attached to a procedure payload."""

    name: str
    kind: str
    config: dict[str, Any] = Field(default_factory=dict)


class ClaimPayload(VTMModel):
    """Payload for a claim memory item."""

    kind: Literal["claim"] = "claim"
    claim: str
    strength: ClaimStrength = ClaimStrength.SUPPORTED


class ProcedurePayload(VTMModel):
    """Payload for an executable or reviewable procedure."""

    kind: Literal["procedure"] = "procedure"
    goal: str
    steps: tuple[ProcedureStep, ...] = Field(default_factory=tuple)
    validator: ValidatorSpec | None = None


class ConstraintPayload(VTMModel):
    """Payload describing a durable constraint or policy."""

    kind: Literal["constraint"] = "constraint"
    statement: str
    severity: str = "info"


class DecisionPayload(VTMModel):
    """Payload describing a recorded decision and rationale."""

    kind: Literal["decision"] = "decision"
    summary: str
    rationale: str | None = None
    supersedes: tuple[str, ...] = Field(default_factory=tuple)


class SummaryCardPayload(VTMModel):
    """Payload for a synthetic summary created from lower-level memory."""

    kind: Literal["summary_card"] = "summary_card"
    summary: str
    detail_level: DetailLevel = DetailLevel.SUMMARY
    supporting_memory_ids: tuple[str, ...] = Field(default_factory=tuple)


MemoryPayload = Annotated[
    ClaimPayload | ProcedurePayload | ConstraintPayload | DecisionPayload | SummaryCardPayload,
    Field(discriminator="kind"),
]


class VisibilityScope(VTMModel):
    """Namespace that determines where a memory item is visible."""

    kind: ScopeKind
    scope_id: str


class ValidityState(VTMModel):
    """Verification state attached to a memory item."""

    status: ValidityStatus = ValidityStatus.PENDING
    dependency_fingerprint: DependencyFingerprint | None = None
    checked_at: datetime | None = None
    reason: str | None = None


class LineageEdge(VTMModel):
    """Directed relationship between two memory items."""

    parent_id: str
    child_id: str
    edge_type: str
    tx_id: str | None = None
    created_at: datetime = Field(default_factory=utc_now)


class MemoryStats(VTMModel):
    """Derived usage counters maintained by retrieval and verification flows."""

    retrieval_count: int = Field(default=0, ge=0)
    verification_count: int = Field(default=0, ge=0)
    last_retrieved_at: datetime | None = None
    last_verified_at: datetime | None = None


RETRIEVABLE_KINDS = {
    MemoryKind.CLAIM,
    MemoryKind.PROCEDURE,
    MemoryKind.CONSTRAINT,
    MemoryKind.DECISION,
}
VERIFIED_STATUSES = {ValidityStatus.VERIFIED, ValidityStatus.RELOCATED}


class MemoryItem(VTMModel):
    """Canonical durable memory record stored by the kernel."""

    memory_id: str = Field(default_factory=new_memory_id)
    kind: MemoryKind
    title: str
    summary: str
    payload: MemoryPayload
    evidence: tuple[EvidenceRef, ...] = Field(default_factory=tuple)
    tags: tuple[str, ...] = Field(default_factory=tuple)
    visibility: VisibilityScope
    validity: ValidityState = Field(default_factory=ValidityState)
    lineage: tuple[LineageEdge, ...] = Field(default_factory=tuple)
    stats: MemoryStats = Field(default_factory=MemoryStats)
    tx_id: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_invariants(self) -> MemoryItem:
        """Enforce cross-field invariants for durable memory records."""
        if self.kind.value != self.payload.kind:
            raise ValueError("memory kind must match payload kind")

        if (
            self.validity.status in VERIFIED_STATUSES
            and self.validity.dependency_fingerprint is None
        ):
            raise ValueError("verified and relocated memories require a dependency fingerprint")

        if (
            self.kind in RETRIEVABLE_KINDS
            and self.validity.status in VERIFIED_STATUSES
            and not self.evidence
        ):
            raise ValueError("verified or relocated claim-like memories require evidence")

        if self.kind is MemoryKind.SUMMARY_CARD:
            if not any(
                evidence.kind in {EvidenceKind.ARTIFACT, EvidenceKind.MEMORY}
                for evidence in self.evidence
            ):
                raise ValueError(
                    "summary cards must reference raw artifacts or lower-level memories"
                )

        return self
