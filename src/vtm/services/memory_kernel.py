"""Primary kernel facade that composes stores, adapters, and services."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, cast

from vtm.anchors import AnchorAdapter, CodeAnchor
from vtm.artifacts import ArtifactRecord
from vtm.cache import CacheEntry, CacheKey
from vtm.evidence import EvidenceRef
from vtm.fingerprints import DependencyFingerprint
from vtm.memory_items import MemoryItem, VisibilityScope
from vtm.retrieval import RetrieveRequest, RetrieveResult
from vtm.services.kernel_artifacts import ArtifactKernelOps
from vtm.services.kernel_mutations import MetadataMutationRunner
from vtm.services.kernel_retrieval import RetrievalKernelOps
from vtm.services.kernel_transactions import TransactionKernelOps
from vtm.services.kernel_validation import ValidationKernelOps
from vtm.services.procedures import CommandProcedureValidator, ProcedureValidator
from vtm.services.retriever import Retriever
from vtm.services.verifier import Verifier
from vtm.stores.base import ArtifactStore, CacheStore, EventStore, MetadataStore
from vtm.stores.sqlite_store import SqliteMetadataStore
from vtm.transactions import TransactionRecord
from vtm.verification import ProcedureValidationResult, VerificationResult


class MemoryKernel(Protocol):
    """Public transactional-memory contract implemented by the kernel facade."""

    def begin_transaction(
        self,
        visibility: VisibilityScope,
        *,
        parent_tx_id: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> TransactionRecord: ...

    def stage_memory_item(self, tx_id: str, item: MemoryItem) -> MemoryItem: ...

    def list_visible_memory(self, tx_id: str) -> tuple[MemoryItem, ...]: ...

    def commit_transaction(self, tx_id: str) -> TransactionRecord: ...

    def rollback_transaction(self, tx_id: str) -> TransactionRecord: ...

    def verify_memory(
        self,
        memory_id: str,
        current_dependency: DependencyFingerprint,
    ) -> tuple[MemoryItem, VerificationResult]: ...

    def validate_procedure(
        self,
        memory_id: str,
        *,
        repo_root: str | None = None,
    ) -> ProcedureValidationResult: ...

    def promote_to_procedure(
        self,
        source_memory_ids: tuple[str, ...],
        procedure: MemoryItem,
    ) -> MemoryItem: ...

    def retrieve(self, request: RetrieveRequest) -> RetrieveResult: ...

    def expand(self, memory_id: str) -> tuple[EvidenceRef, ...]: ...

    def build_code_anchor(self, source_path: str, symbol: str) -> CodeAnchor: ...

    def capture_artifact(
        self,
        data: bytes,
        *,
        content_type: str = "application/octet-stream",
        tool_name: str | None = None,
        tool_version: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> ArtifactRecord: ...

    def artifact_evidence(
        self,
        record: ArtifactRecord,
        *,
        label: str | None = None,
        summary: str | None = None,
    ) -> EvidenceRef: ...

    def anchor_evidence(
        self,
        anchor: CodeAnchor,
        *,
        label: str | None = None,
        summary: str | None = None,
    ) -> EvidenceRef: ...

    def save_cache_entry(self, entry: CacheEntry) -> None: ...

    def get_cache_entry(self, key: CacheKey) -> CacheEntry | None: ...


class CodeAnchorBuilder(Protocol):
    """Minimal contract for building code anchors from repository symbols."""

    def build_anchor(self, source_path: str, symbol: str) -> CodeAnchor: ...


class TransactionalMemoryKernel:
    """Reference kernel facade used by applications, harnesses, and benchmarks."""

    def __init__(
        self,
        *,
        metadata_store: MetadataStore,
        event_store: EventStore,
        artifact_store: ArtifactStore,
        cache_store: CacheStore,
        verifier: Verifier,
        retriever: Retriever,
        anchor_adapter: AnchorAdapter | None = None,
        anchor_builder: CodeAnchorBuilder | None = None,
        procedure_validator: ProcedureValidator | None = None,
        require_shared_event_store: bool = True,
    ) -> None:
        """Compose storage and service collaborators into one kernel instance."""
        shared_sqlite_event_store = cast(object, event_store) is cast(
            object,
            metadata_store,
        ) and isinstance(metadata_store, SqliteMetadataStore)
        if require_shared_event_store and not shared_sqlite_event_store:
            raise ValueError(
                "atomic event semantics require event_store to be the same "
                "SqliteMetadataStore instance as metadata_store; set "
                "require_shared_event_store=False to opt into degraded semantics"
            )

        resolved_anchor_builder = anchor_adapter or anchor_builder
        mutations = MetadataMutationRunner(
            metadata_store=metadata_store,
            event_store=event_store,
        )
        resolved_procedure_validator = procedure_validator or CommandProcedureValidator(
            artifact_store,
        )

        self._validation = ValidationKernelOps(
            metadata_store=metadata_store,
            artifact_store=artifact_store,
            verifier=verifier,
            procedure_validator=resolved_procedure_validator,
            mutations=mutations,
        )
        self._transactions = TransactionKernelOps(
            metadata_store=metadata_store,
            mutations=mutations,
            validate_committable_item=self._validation.validate_committable_item,
        )
        self._artifacts = ArtifactKernelOps(
            event_store=event_store,
            artifact_store=artifact_store,
            anchor_builder=resolved_anchor_builder,
        )
        self._retrieval_ops = RetrievalKernelOps(
            metadata_store=metadata_store,
            event_store=event_store,
            cache_store=cache_store,
            retriever=retriever,
            mutations=mutations,
            verify_memory=self._validation.verify_memory,
        )

    def begin_transaction(
        self,
        visibility: VisibilityScope,
        *,
        parent_tx_id: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> TransactionRecord:
        """Open a new transaction scoped to the provided visibility boundary."""
        return self._transactions.begin_transaction(
            visibility,
            parent_tx_id=parent_tx_id,
            metadata=metadata,
        )

    def stage_memory_item(self, tx_id: str, item: MemoryItem) -> MemoryItem:
        """Stage a candidate memory item inside an active transaction."""
        return self._transactions.stage_memory_item(tx_id, item)

    def list_visible_memory(self, tx_id: str) -> tuple[MemoryItem, ...]:
        """List committed plus staged memory visible to the given transaction."""
        return self._transactions.list_visible_memory(tx_id)

    def commit_transaction(self, tx_id: str) -> TransactionRecord:
        """Commit all staged memory for the transaction."""
        return self._transactions.commit_transaction(tx_id)

    def rollback_transaction(self, tx_id: str) -> TransactionRecord:
        """Roll back an active transaction and clear staged memory."""
        return self._transactions.rollback_transaction(tx_id)

    def verify_memory(
        self,
        memory_id: str,
        current_dependency: DependencyFingerprint,
    ) -> tuple[MemoryItem, VerificationResult]:
        """Re-check a memory item against the current dependency fingerprint."""
        return self._validation.verify_memory(memory_id, current_dependency)

    def validate_procedure(
        self,
        memory_id: str,
        *,
        repo_root: str | None = None,
    ) -> ProcedureValidationResult:
        """Execute the configured validator for a procedure memory."""
        return self._validation.validate_procedure(memory_id, repo_root=repo_root)

    def promote_to_procedure(
        self,
        source_memory_ids: tuple[str, ...],
        procedure: MemoryItem,
    ) -> MemoryItem:
        """Write a procedure memory derived from existing source memories."""
        return self._validation.promote_to_procedure(source_memory_ids, procedure)

    def retrieve(self, request: RetrieveRequest) -> RetrieveResult:
        """Retrieve visible committed memory using the configured retriever."""
        return self._retrieval_ops.retrieve(request)

    def expand(self, memory_id: str) -> tuple[EvidenceRef, ...]:
        """Return raw evidence attached to a stored memory item."""
        return self._retrieval_ops.expand(memory_id)

    def build_code_anchor(self, source_path: str, symbol: str) -> CodeAnchor:
        """Build a code anchor for a symbol in the current repository state."""
        return self._artifacts.build_code_anchor(source_path, symbol)

    def capture_artifact(
        self,
        data: bytes,
        *,
        content_type: str = "application/octet-stream",
        tool_name: str | None = None,
        tool_version: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> ArtifactRecord:
        """Persist an artifact blob and commit the capture immediately."""
        return self._artifacts.capture_artifact(
            data,
            content_type=content_type,
            tool_name=tool_name,
            tool_version=tool_version,
            metadata=metadata,
        )

    def artifact_evidence(
        self,
        record: ArtifactRecord,
        *,
        label: str | None = None,
        summary: str | None = None,
    ) -> EvidenceRef:
        """Convert an artifact record into an evidence reference."""
        return self._artifacts.artifact_evidence(record, label=label, summary=summary)

    def anchor_evidence(
        self,
        anchor: CodeAnchor,
        *,
        label: str | None = None,
        summary: str | None = None,
    ) -> EvidenceRef:
        """Convert a code anchor into an evidence reference."""
        return self._artifacts.anchor_evidence(anchor, label=label, summary=summary)

    def save_cache_entry(self, entry: CacheEntry) -> None:
        """Persist a cache entry through the retrieval subsystem."""
        self._retrieval_ops.save_cache_entry(entry)

    def get_cache_entry(self, key: CacheKey) -> CacheEntry | None:
        """Load a cache entry if it exists and is still valid."""
        return self._retrieval_ops.get_cache_entry(key)
