"""Kernel-side transaction helpers for staging and committing memory."""

from __future__ import annotations

from collections.abc import Callable

from vtm.base import utc_now
from vtm.enums import TxState
from vtm.events import MemoryEvent
from vtm.memory_items import LineageEdge, MemoryItem, VisibilityScope
from vtm.services.kernel_mutations import MetadataMutationRunner
from vtm.stores.base import MetadataStore
from vtm.transactions import TransactionRecord


class TransactionKernelOps:
    """Owns transaction lifecycle, staging, visibility, and commit flows."""

    def __init__(
        self,
        *,
        metadata_store: MetadataStore,
        mutations: MetadataMutationRunner,
        validate_committable_item: Callable[[MemoryItem], None],
    ) -> None:
        """Create transaction helpers around metadata and mutation collaborators."""
        self._metadata_store = metadata_store
        self._mutations = mutations
        self._validate_committable_item = validate_committable_item

    def begin_transaction(
        self,
        visibility: VisibilityScope,
        *,
        parent_tx_id: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> TransactionRecord:
        """Open a new active transaction."""
        if parent_tx_id is not None:
            parent = self._require_transaction(parent_tx_id)
            if parent.state is not TxState.ACTIVE:
                raise ValueError(f"parent transaction is not active: {parent_tx_id}")
        transaction = TransactionRecord(
            parent_tx_id=parent_tx_id,
            visibility=visibility,
            metadata=dict(metadata or {}),
        )
        return self._mutations.run(
            lambda: self._persist_transaction_begin(transaction),
            build_events=lambda opened: (
                MemoryEvent(
                    event_type="tx_begin",
                    tx_id=opened.tx_id,
                    payload={"parent_tx_id": parent_tx_id, "scope_id": visibility.scope_id},
                ),
            ),
        )

    def stage_memory_item(self, tx_id: str, item: MemoryItem) -> MemoryItem:
        """Stage a memory item inside an active transaction."""
        transaction = self._require_active_transaction(tx_id)
        staged_item = item.model_copy(update={"tx_id": tx_id, "updated_at": utc_now()})
        return self._mutations.run(
            lambda: self._persist_staged_item(tx_id, staged_item),
            build_events=lambda persisted: (
                MemoryEvent(
                    event_type="memory_staged",
                    tx_id=tx_id,
                    memory_id=persisted.memory_id,
                    payload={
                        "scope_id": transaction.visibility.scope_id,
                        "kind": persisted.kind.value,
                    },
                ),
            ),
        )

    def list_visible_memory(self, tx_id: str) -> tuple[MemoryItem, ...]:
        """Return committed plus ancestor-staged memory visible to the tx."""
        transaction = self._require_transaction(tx_id)
        visible = {
            item.memory_id: item
            for item in self._metadata_store.query_memory_items([transaction.visibility])
        }
        for ancestor_tx_id in self._ancestor_chain(tx_id):
            for item in self._metadata_store.list_staged_memory_items(ancestor_tx_id):
                visible[item.memory_id] = item
        return tuple(sorted(visible.values(), key=lambda item: item.memory_id))

    def commit_transaction(self, tx_id: str) -> TransactionRecord:
        """Commit an active transaction, merging into the parent when present."""
        transaction = self._require_active_transaction(tx_id)
        staged_items = list(self._metadata_store.list_staged_memory_items(tx_id))

        if transaction.parent_tx_id is not None:
            parent = self._require_active_transaction(transaction.parent_tx_id)
            committed = transaction.model_copy(
                update={"state": TxState.COMMITTED, "committed_at": utc_now()}
            )
            return self._mutations.run(
                lambda: self._persist_child_commit(tx_id, parent.tx_id, committed),
                build_events=lambda result: (
                    MemoryEvent(
                        event_type="tx_commit_merged",
                        tx_id=tx_id,
                        payload={"parent_tx_id": parent.tx_id, "memory_count": len(staged_items)},
                    ),
                ),
            )

        for item in staged_items:
            self._validate_committable_item(item)

        committed = transaction.model_copy(
            update={"state": TxState.COMMITTED, "committed_at": utc_now()}
        )
        return self._mutations.run(
            lambda: self._persist_root_commit(tx_id, staged_items, committed),
            build_events=lambda result: (
                MemoryEvent(
                    event_type="tx_commit",
                    tx_id=tx_id,
                    payload={"memory_count": len(staged_items)},
                ),
            ),
        )

    def rollback_transaction(self, tx_id: str) -> TransactionRecord:
        """Roll back an active transaction and discard staged items."""
        transaction = self._require_active_transaction(tx_id)
        discarded = len(self._metadata_store.list_staged_memory_items(tx_id))
        rolled_back = transaction.model_copy(
            update={"state": TxState.ROLLED_BACK, "rolled_back_at": utc_now()}
        )
        return self._mutations.run(
            lambda: self._persist_rollback(tx_id, rolled_back),
            build_events=lambda result: (
                MemoryEvent(
                    event_type="tx_rollback",
                    tx_id=tx_id,
                    payload={"discarded_memory_count": discarded},
                ),
            ),
        )

    def _persist_transaction_begin(self, transaction: TransactionRecord) -> TransactionRecord:
        self._metadata_store.save_transaction(transaction)
        return transaction

    def _persist_staged_item(self, tx_id: str, item: MemoryItem) -> MemoryItem:
        self._metadata_store.append_staged_memory_item(tx_id, item)
        return item

    def _persist_child_commit(
        self,
        tx_id: str,
        parent_tx_id: str,
        committed: TransactionRecord,
    ) -> TransactionRecord:
        self._metadata_store.move_staged_memory_items(tx_id, parent_tx_id)
        self._metadata_store.save_transaction(committed)
        return committed

    def _persist_root_commit(
        self,
        tx_id: str,
        staged_items: list[MemoryItem],
        committed: TransactionRecord,
    ) -> TransactionRecord:
        for item in staged_items:
            self._metadata_store.save_memory_item(item)
            self._metadata_store.save_lineage_edge(
                LineageEdge(
                    parent_id=item.tx_id or tx_id,
                    child_id=item.memory_id,
                    edge_type="committed_in",
                    tx_id=tx_id,
                )
            )
        self._metadata_store.clear_staged_memory_items(tx_id)
        self._metadata_store.save_transaction(committed)
        return committed

    def _persist_rollback(self, tx_id: str, rolled_back: TransactionRecord) -> TransactionRecord:
        self._metadata_store.clear_staged_memory_items(tx_id)
        self._metadata_store.save_transaction(rolled_back)
        return rolled_back

    def _require_transaction(self, tx_id: str) -> TransactionRecord:
        transaction = self._metadata_store.get_transaction(tx_id)
        if transaction is None:
            raise KeyError(f"unknown transaction: {tx_id}")
        return transaction

    def _require_active_transaction(self, tx_id: str) -> TransactionRecord:
        transaction = self._require_transaction(tx_id)
        if transaction.state is not TxState.ACTIVE:
            raise ValueError(f"transaction is not active: {tx_id}")
        return transaction

    def _ancestor_chain(self, tx_id: str) -> tuple[str, ...]:
        ordered: list[str] = []
        current = self._require_transaction(tx_id)
        while True:
            ordered.append(current.tx_id)
            if current.parent_tx_id is None:
                break
            current = self._require_transaction(current.parent_tx_id)
        ordered.reverse()
        return tuple(ordered)
