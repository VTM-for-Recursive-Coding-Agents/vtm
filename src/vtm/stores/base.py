"""Storage protocols implemented by the concrete VTM stores."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from typing import Any, Protocol, TypeVar

from vtm.artifacts import ArtifactIntegrityReport, ArtifactRecord, ArtifactRepairReport
from vtm.cache import CacheEntry, CacheKey
from vtm.embeddings import EmbeddingIndexEntry
from vtm.enums import ArtifactCaptureState, ValidityStatus
from vtm.events import MemoryEvent
from vtm.memory_items import LineageEdge, MemoryItem, VisibilityScope
from vtm.transactions import TransactionRecord

ResultT = TypeVar("ResultT")


class MetadataStore(Protocol):
    """Persists durable memory, lineage, transaction, and staging state."""

    def save_memory_item(self, item: MemoryItem) -> None: ...

    def get_memory_item(self, memory_id: str) -> MemoryItem | None: ...

    def list_memory_items(self) -> Sequence[MemoryItem]: ...

    def query_memory_items(
        self,
        scopes: Sequence[VisibilityScope],
        statuses: Sequence[ValidityStatus] | None = None,
        allow_quarantined: bool = False,
    ) -> Sequence[MemoryItem]: ...

    def save_lineage_edge(self, edge: LineageEdge) -> None: ...

    def list_lineage_edges(
        self,
        *,
        child_id: str | None = None,
        tx_id: str | None = None,
    ) -> Sequence[LineageEdge]: ...

    def save_transaction(self, transaction: TransactionRecord) -> None: ...

    def get_transaction(self, tx_id: str) -> TransactionRecord | None: ...

    def list_transactions(self) -> Sequence[TransactionRecord]: ...

    def append_staged_memory_item(self, tx_id: str, item: MemoryItem) -> None: ...

    def list_staged_memory_items(self, tx_id: str) -> Sequence[MemoryItem]: ...

    def move_staged_memory_items(self, source_tx_id: str, target_tx_id: str) -> None: ...

    def clear_staged_memory_items(self, tx_id: str) -> None: ...

    def run_atomically(self, operation: Callable[[], ResultT]) -> ResultT: ...


class EventStore(Protocol):
    """Append-only event ledger for memory-side mutations."""

    def save_event(self, event: MemoryEvent) -> None: ...

    def get_event(self, event_id: str) -> MemoryEvent | None: ...

    def list_events(self) -> Sequence[MemoryEvent]: ...


class ArtifactStore(Protocol):
    """Captures immutable blobs plus lifecycle metadata."""

    def prepare_bytes(
        self,
        data: bytes,
        *,
        content_type: str = "application/octet-stream",
        tool_name: str | None = None,
        tool_version: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        artifact_id: str | None = None,
        capture_group_id: str | None = None,
        actor: str = "system",
        session_id: str | None = None,
    ) -> ArtifactRecord: ...

    def commit_artifact(self, artifact_id: str) -> ArtifactRecord: ...

    def abandon_artifact(
        self,
        artifact_id: str,
        *,
        reason: str | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> ArtifactRecord: ...

    def put_bytes(
        self,
        data: bytes,
        *,
        content_type: str = "application/octet-stream",
        tool_name: str | None = None,
        tool_version: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        artifact_id: str | None = None,
    ) -> ArtifactRecord: ...

    def get_artifact_record_by_id(self, artifact_id: str) -> ArtifactRecord | None: ...

    def get_artifact_record_by_sha256(self, sha256: str) -> ArtifactRecord | None: ...

    def list_artifact_records_by_sha256(self, sha256: str) -> Sequence[ArtifactRecord]: ...

    def list_artifact_records(
        self,
        *,
        capture_state: ArtifactCaptureState | None = None,
    ) -> Sequence[ArtifactRecord]: ...

    def audit_integrity(self) -> ArtifactIntegrityReport: ...

    def read_bytes_by_id(self, artifact_id: str) -> bytes | None: ...

    def abandon_stale_prepared_artifacts(self) -> Sequence[ArtifactRecord]: ...

    def cleanup_orphaned_blobs(self) -> Sequence[str]: ...

    def repair_integrity(self) -> ArtifactRepairReport: ...


class CacheStore(Protocol):
    """Persists deterministic cache entries keyed by repo and env state."""

    def save_cache_entry(self, entry: CacheEntry) -> None: ...

    def get_cache_entry(
        self,
        key: CacheKey,
        *,
        now: datetime | None = None,
    ) -> CacheEntry | None: ...

    def delete_cache_entry(self, key: CacheKey) -> None: ...

    def list_cache_entries(self) -> Sequence[CacheEntry]: ...


class EmbeddingIndexStore(Protocol):
    """Stores derived embedding vectors for retrieval adapters."""

    def save_entry(self, entry: EmbeddingIndexEntry) -> None: ...

    def get_entry(self, memory_id: str, adapter_id: str) -> EmbeddingIndexEntry | None: ...

    def list_entries(
        self,
        *,
        adapter_id: str | None = None,
    ) -> Sequence[EmbeddingIndexEntry]: ...

    def delete_entry(self, memory_id: str, adapter_id: str) -> None: ...
