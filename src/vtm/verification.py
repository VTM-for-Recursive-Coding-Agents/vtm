"""Result models for verification and procedure validation."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import Field

from vtm.anchors import AnchorRelocation
from vtm.base import VTMModel, utc_now
from vtm.enums import ValidityStatus
from vtm.evidence import EvidenceRef
from vtm.memory_items import ValidatorSpec, ValidityState


class VerificationResult(VTMModel):
    """Outcome of checking a memory item against current dependencies."""

    memory_id: str
    previous_status: ValidityStatus
    current_status: ValidityStatus
    dependency_changed: bool
    checked_at: datetime = Field(default_factory=utc_now)
    reasons: tuple[str, ...] = Field(default_factory=tuple)
    updated_validity: ValidityState
    relocation: AnchorRelocation | None = None
    updated_evidence: tuple[EvidenceRef, ...] | None = None
    skipped: bool = False


class ProcedureValidationResult(VTMModel):
    """Outcome of executing a procedure validator."""

    memory_id: str
    validator_spec: ValidatorSpec
    success: bool
    exit_code: int | None = None
    stdout_artifact_id: str
    stderr_artifact_id: str
    checked_at: datetime = Field(default_factory=utc_now)
    status: ValidityStatus
    reason: str
    metadata: dict[str, Any] = Field(default_factory=dict)
