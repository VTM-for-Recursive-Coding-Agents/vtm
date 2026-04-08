"""Evidence reference records stored on memory items."""

from __future__ import annotations

from pydantic import model_validator

from vtm.anchors import CodeAnchor
from vtm.base import VTMModel
from vtm.enums import EvidenceKind


class ArtifactRef(VTMModel):
    """Minimal pointer to an artifact record and blob digest."""

    artifact_id: str
    sha256: str
    content_type: str | None = None


class EvidenceRef(VTMModel):
    """Typed reference from a memory item to supporting evidence."""

    kind: EvidenceKind
    ref_id: str
    artifact_ref: ArtifactRef | None = None
    code_anchor: CodeAnchor | None = None
    memory_id: str | None = None
    label: str | None = None
    summary: str | None = None

    @model_validator(mode="after")
    def validate_target(self) -> EvidenceRef:
        """Enforce that the populated target matches the evidence kind."""
        if self.kind is EvidenceKind.ARTIFACT:
            if (
                self.artifact_ref is None
                or self.code_anchor is not None
                or self.memory_id is not None
            ):
                raise ValueError("artifact evidence must include only artifact_ref")
        elif self.kind is EvidenceKind.CODE_ANCHOR:
            if (
                self.code_anchor is None
                or self.artifact_ref is not None
                or self.memory_id is not None
            ):
                raise ValueError("code_anchor evidence must include only code_anchor")
        elif self.kind is EvidenceKind.MEMORY:
            if (
                self.memory_id is None
                or self.artifact_ref is not None
                or self.code_anchor is not None
            ):
                raise ValueError("memory evidence must include only memory_id")
        return self
