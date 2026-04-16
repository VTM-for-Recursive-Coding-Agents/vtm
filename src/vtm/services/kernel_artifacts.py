"""Artifact and code-anchor helpers used by the kernel facade."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import suppress
from typing import Protocol
from uuid import uuid4

from vtm.anchors import CodeAnchor
from vtm.artifacts import ArtifactRecord
from vtm.enums import EvidenceKind
from vtm.events import MemoryEvent
from vtm.evidence import ArtifactRef, EvidenceRef
from vtm.stores.base import ArtifactStore, EventStore


class CodeAnchorBuilder(Protocol):
    """Minimal contract for building code anchors inside the kernel."""

    def build_anchor(self, source_path: str, symbol: str) -> CodeAnchor: ...


class ArtifactKernelOps:
    """Owns artifact capture, anchor building, and evidence conversion."""

    def __init__(
        self,
        *,
        event_store: EventStore,
        artifact_store: ArtifactStore,
        anchor_builder: CodeAnchorBuilder | None,
    ) -> None:
        """Create the artifact helper with its backing stores and builder."""
        self._event_store = event_store
        self._artifact_store = artifact_store
        self._anchor_builder = anchor_builder

    def build_code_anchor(self, source_path: str, symbol: str) -> CodeAnchor:
        """Build a code anchor and emit an audit event."""
        if self._anchor_builder is None:
            raise RuntimeError("no code anchor builder configured")
        anchor = self._anchor_builder.build_anchor(source_path, symbol)
        self._event_store.save_event(
            MemoryEvent(
                event_type="anchor_built",
                tool_name="anchor-adapter",
                payload={"path": anchor.path, "symbol": anchor.symbol},
            )
        )
        return anchor

    def capture_artifact(
        self,
        data: bytes,
        *,
        content_type: str = "application/octet-stream",
        tool_name: str | None = None,
        tool_version: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> ArtifactRecord:
        """Capture an artifact through prepare-and-commit semantics."""
        capture_group_id = f"capgrp_{uuid4().hex}"
        prepared = None
        record = None
        try:
            prepared = self._artifact_store.prepare_bytes(
                data,
                content_type=content_type,
                tool_name=tool_name,
                tool_version=tool_version,
                metadata=metadata,
                capture_group_id=capture_group_id,
                actor="kernel",
            )
            self._event_store.save_event(
                MemoryEvent(
                    event_type="artifact_capture_prepared",
                    payload={
                        "artifact_id": prepared.artifact_id,
                        "capture_group_id": capture_group_id,
                        "sha256": prepared.sha256,
                        "tool_name": prepared.tool_name,
                        "capture_state": prepared.capture_state.value,
                    },
                )
            )
            record = self._artifact_store.commit_artifact(prepared.artifact_id)
            self._event_store.save_event(
                MemoryEvent(
                    event_type="artifact_captured",
                    tool_name=tool_name,
                    payload={
                        "artifact_id": record.artifact_id,
                        "capture_group_id": capture_group_id,
                        "sha256": record.sha256,
                        "tool_name": record.tool_name,
                        "capture_state": record.capture_state.value,
                    },
                )
            )
            return record
        except Exception:
            artifact_id = record.artifact_id if record is not None else None
            if artifact_id is None and prepared is not None:
                artifact_id = prepared.artifact_id
            if artifact_id is not None:
                self._best_effort_abandon(
                    artifact_id,
                    reason="artifact_capture_writeback_failed",
                )
            raise

    def artifact_evidence(
        self,
        record: ArtifactRecord,
        *,
        label: str | None = None,
        summary: str | None = None,
    ) -> EvidenceRef:
        """Convert an artifact record into an evidence reference."""
        return EvidenceRef(
            kind=EvidenceKind.ARTIFACT,
            ref_id=f"artifact:{record.artifact_id}",
            artifact_ref=ArtifactRef(
                artifact_id=record.artifact_id,
                sha256=record.sha256,
                content_type=record.content_type,
            ),
            label=label,
            summary=summary,
        )

    def anchor_evidence(
        self,
        anchor: CodeAnchor,
        *,
        label: str | None = None,
        summary: str | None = None,
    ) -> EvidenceRef:
        """Convert a code anchor into an evidence reference."""
        return EvidenceRef(
            kind=EvidenceKind.CODE_ANCHOR,
            ref_id=f"anchor:{anchor.path}:{anchor.symbol}",
            code_anchor=anchor,
            label=label,
            summary=summary,
        )

    def _best_effort_abandon(self, artifact_id: str, *, reason: str) -> None:
        with suppress(Exception):
            self._artifact_store.abandon_artifact(
                artifact_id,
                reason=reason,
                provenance={
                    "origin": "kernel_artifact_capture",
                    "stage": "event_writeback",
                },
            )
