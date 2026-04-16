"""Artifact capture records for durable tool-output storage."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import Field

from vtm.base import VTMModel, utc_now
from vtm.enums import ArtifactCaptureState
from vtm.ids import new_artifact_id


class ArtifactRecord(VTMModel):
    """Metadata describing a captured artifact blob."""

    artifact_id: str = Field(default_factory=new_artifact_id)
    sha256: str
    relative_path: str
    size_bytes: int = Field(ge=0)
    content_type: str = "application/octet-stream"
    tool_name: str | None = None
    tool_version: str | None = None
    capture_state: ArtifactCaptureState = ArtifactCaptureState.COMMITTED
    capture_group_id: str | None = None
    actor: str = "system"
    session_id: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    committed_at: datetime | None = None
    abandoned_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ArtifactIntegrityReport(VTMModel):
    """Inspectable integrity issues found in the artifact store."""

    prepared_artifact_ids: tuple[str, ...] = Field(default_factory=tuple)
    committed_missing_blob_artifact_ids: tuple[str, ...] = Field(default_factory=tuple)
    orphaned_blob_paths: tuple[str, ...] = Field(default_factory=tuple)
    abandoned_artifact_ids_by_reason: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    abandoned_artifact_ids_by_origin: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    abandoned_artifact_ids_without_reason: tuple[str, ...] = Field(default_factory=tuple)


class ArtifactRepairReport(VTMModel):
    """Outcome of applying safe repair actions to the artifact store."""

    audit_before: ArtifactIntegrityReport
    audit_after: ArtifactIntegrityReport
    abandoned_prepared_artifact_ids: tuple[str, ...] = Field(default_factory=tuple)
    removed_orphaned_blob_paths: tuple[str, ...] = Field(default_factory=tuple)
    unresolved_committed_missing_blob_artifact_ids: tuple[str, ...] = Field(
        default_factory=tuple
    )
