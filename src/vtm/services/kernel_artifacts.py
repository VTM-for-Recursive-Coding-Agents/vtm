from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol
from uuid import uuid4

from vtm.anchors import CodeAnchor
from vtm.artifacts import ArtifactRecord
from vtm.enums import EvidenceKind
from vtm.events import MemoryEvent
from vtm.evidence import ArtifactRef, EvidenceRef
from vtm.stores.base import ArtifactStore, EventStore


class CodeAnchorBuilder(Protocol):
    def build_anchor(self, source_path: str, symbol: str) -> CodeAnchor: ...


class ArtifactKernelOps:
    def __init__(
        self,
        *,
        event_store: EventStore,
        artifact_store: ArtifactStore,
        anchor_builder: CodeAnchorBuilder | None,
    ) -> None:
        self._event_store = event_store
        self._artifact_store = artifact_store
        self._anchor_builder = anchor_builder

    def build_code_anchor(self, source_path: str, symbol: str) -> CodeAnchor:
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
        capture_group_id = f"capgrp_{uuid4().hex}"
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

    def artifact_evidence(
        self,
        record: ArtifactRecord,
        *,
        label: str | None = None,
        summary: str | None = None,
    ) -> EvidenceRef:
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
        return EvidenceRef(
            kind=EvidenceKind.CODE_ANCHOR,
            ref_id=f"anchor:{anchor.path}:{anchor.symbol}",
            code_anchor=anchor,
            label=label,
            summary=summary,
        )
