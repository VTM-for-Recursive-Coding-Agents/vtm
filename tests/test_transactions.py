from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from vtm.adapters.python_ast import PythonAstSyntaxAdapter
from vtm.adapters.tree_sitter import PythonTreeSitterSyntaxAdapter
from vtm.enums import TxState
from vtm.memory_items import LineageEdge
from vtm.services.memory_kernel import TransactionalMemoryKernel
from vtm.services.procedures import CommandProcedureValidator
from vtm.services.retriever import LexicalRetriever
from vtm.services.verifier import BasicVerifier
from vtm.stores.artifact_store import FilesystemArtifactStore
from vtm.stores.cache_store import SqliteCacheStore
from vtm.stores.sqlite_store import SqliteMetadataStore
from vtm.transactions import TransactionRecord


def _open_kernel(
    tmp_path: Path,
    *,
    metadata_store_factory: Callable[[Path], SqliteMetadataStore] | None = None,
) -> tuple[
    TransactionalMemoryKernel,
    SqliteMetadataStore,
    FilesystemArtifactStore,
    SqliteCacheStore,
]:
    metadata_store = (
        metadata_store_factory(tmp_path)
        if metadata_store_factory is not None
        else SqliteMetadataStore(
            tmp_path / "metadata.sqlite",
            event_log_path=tmp_path / "events.jsonl",
        )
    )
    artifact_store = FilesystemArtifactStore(tmp_path / "artifacts")
    cache_store = SqliteCacheStore(tmp_path / "cache.sqlite", event_store=metadata_store)
    anchor_builder = PythonTreeSitterSyntaxAdapter(fallback=PythonAstSyntaxAdapter())
    kernel = TransactionalMemoryKernel(
        metadata_store=metadata_store,
        event_store=metadata_store,
        artifact_store=artifact_store,
        cache_store=cache_store,
        verifier=BasicVerifier(relocator=anchor_builder),
        retriever=LexicalRetriever(metadata_store),
        anchor_adapter=anchor_builder,
        procedure_validator=CommandProcedureValidator(artifact_store),
    )
    return kernel, metadata_store, artifact_store, cache_store


def _close_kernel(
    metadata_store: SqliteMetadataStore,
    artifact_store: FilesystemArtifactStore,
    cache_store: SqliteCacheStore,
) -> None:
    cache_store.close()
    artifact_store.close()
    metadata_store.close()


def test_nested_commit_merges_then_persists(kernel, metadata_store, memory_factory, scope) -> None:
    root = kernel.begin_transaction(scope)
    child = kernel.begin_transaction(scope, parent_tx_id=root.tx_id)

    root_item = kernel.stage_memory_item(root.tx_id, memory_factory(title="Root memory"))
    child_item = kernel.stage_memory_item(child.tx_id, memory_factory(title="Child memory"))

    child_commit = kernel.commit_transaction(child.tx_id)
    assert child_commit.state is TxState.COMMITTED
    assert metadata_store.list_memory_items() == ()

    visible_in_root = {item.memory_id for item in kernel.list_visible_memory(root.tx_id)}
    assert visible_in_root == {root_item.memory_id, child_item.memory_id}

    root_commit = kernel.commit_transaction(root.tx_id)
    assert root_commit.state is TxState.COMMITTED

    persisted = metadata_store.list_memory_items()
    assert {item.memory_id for item in persisted} == {root_item.memory_id, child_item.memory_id}

    lineage_edges = metadata_store.list_lineage_edges(tx_id=root.tx_id)
    assert {edge.child_id for edge in lineage_edges} == {root_item.memory_id, child_item.memory_id}


def test_rollback_discards_staged_writes_but_keeps_audit_log(
    kernel,
    metadata_store,
    memory_factory,
    scope,
) -> None:
    tx = kernel.begin_transaction(scope)
    staged = kernel.stage_memory_item(tx.tx_id, memory_factory(title="Discarded memory"))

    rolled_back = kernel.rollback_transaction(tx.tx_id)
    assert rolled_back.state is TxState.ROLLED_BACK
    assert metadata_store.get_memory_item(staged.memory_id) is None
    assert any(event.event_type == "tx_rollback" for event in metadata_store.list_events())


def test_staged_items_remain_visible_after_kernel_restart(
    tmp_path: Path,
    memory_factory,
    scope,
) -> None:
    kernel, metadata_store, artifact_store, cache_store = _open_kernel(tmp_path)
    tx = kernel.begin_transaction(scope)
    staged = kernel.stage_memory_item(tx.tx_id, memory_factory(title="Restart-visible memory"))
    _close_kernel(metadata_store, artifact_store, cache_store)

    (
        reopened_kernel,
        reopened_metadata,
        reopened_artifact_store,
        reopened_cache_store,
    ) = _open_kernel(tmp_path)
    try:
        visible = {item.memory_id for item in reopened_kernel.list_visible_memory(tx.tx_id)}
        assert staged.memory_id in visible
        assert reopened_metadata.get_memory_item(staged.memory_id) is None
    finally:
        _close_kernel(reopened_metadata, reopened_artifact_store, reopened_cache_store)


def test_commit_after_restart_persists_staged_items(
    tmp_path: Path,
    memory_factory,
    scope,
) -> None:
    kernel, metadata_store, artifact_store, cache_store = _open_kernel(tmp_path)
    tx = kernel.begin_transaction(scope)
    staged = kernel.stage_memory_item(tx.tx_id, memory_factory(title="Restart-commit memory"))
    _close_kernel(metadata_store, artifact_store, cache_store)

    (
        reopened_kernel,
        reopened_metadata,
        reopened_artifact_store,
        reopened_cache_store,
    ) = _open_kernel(tmp_path)
    try:
        committed = reopened_kernel.commit_transaction(tx.tx_id)
        assert committed.state is TxState.COMMITTED
        persisted = reopened_metadata.get_memory_item(staged.memory_id)
        assert persisted is not None
        assert persisted.memory_id == staged.memory_id
    finally:
        _close_kernel(reopened_metadata, reopened_artifact_store, reopened_cache_store)


def test_rollback_after_restart_discards_staged_items(
    tmp_path: Path,
    memory_factory,
    scope,
) -> None:
    kernel, metadata_store, artifact_store, cache_store = _open_kernel(tmp_path)
    tx = kernel.begin_transaction(scope)
    staged = kernel.stage_memory_item(tx.tx_id, memory_factory(title="Restart-rollback memory"))
    _close_kernel(metadata_store, artifact_store, cache_store)

    (
        reopened_kernel,
        reopened_metadata,
        reopened_artifact_store,
        reopened_cache_store,
    ) = _open_kernel(tmp_path)
    try:
        rolled_back = reopened_kernel.rollback_transaction(tx.tx_id)
        assert rolled_back.state is TxState.ROLLED_BACK
        assert reopened_metadata.get_memory_item(staged.memory_id) is None
        assert reopened_kernel.list_visible_memory(tx.tx_id) == ()
    finally:
        _close_kernel(reopened_metadata, reopened_artifact_store, reopened_cache_store)


def test_nested_child_merge_remains_visible_after_restart(
    tmp_path: Path,
    memory_factory,
    scope,
) -> None:
    kernel, metadata_store, artifact_store, cache_store = _open_kernel(tmp_path)
    root = kernel.begin_transaction(scope)
    child = kernel.begin_transaction(scope, parent_tx_id=root.tx_id)
    root_item = kernel.stage_memory_item(root.tx_id, memory_factory(title="Root restart memory"))
    child_item = kernel.stage_memory_item(child.tx_id, memory_factory(title="Child restart memory"))
    _close_kernel(metadata_store, artifact_store, cache_store)

    (
        reopened_kernel,
        reopened_metadata,
        reopened_artifact_store,
        reopened_cache_store,
    ) = _open_kernel(tmp_path)
    try:
        child_commit = reopened_kernel.commit_transaction(child.tx_id)
        assert child_commit.state is TxState.COMMITTED

        visible_in_root = {
            item.memory_id for item in reopened_kernel.list_visible_memory(root.tx_id)
        }
        assert visible_in_root == {root_item.memory_id, child_item.memory_id}
    finally:
        _close_kernel(reopened_metadata, reopened_artifact_store, reopened_cache_store)

    (
        restarted_kernel,
        restarted_metadata,
        restarted_artifact_store,
        restarted_cache_store,
    ) = _open_kernel(tmp_path)
    try:
        visible_after_restart = {
            item.memory_id for item in restarted_kernel.list_visible_memory(root.tx_id)
        }
        assert visible_after_restart == {root_item.memory_id, child_item.memory_id}
    finally:
        _close_kernel(restarted_metadata, restarted_artifact_store, restarted_cache_store)


def test_root_commit_rolls_back_metadata_writes_when_persistence_fails(
    tmp_path: Path,
    memory_factory,
    scope,
) -> None:
    class FailingMetadataStore(SqliteMetadataStore):
        def save_lineage_edge(self, edge: LineageEdge) -> None:
            raise RuntimeError("simulated lineage failure")

    def build_metadata_store(base_path: Path) -> SqliteMetadataStore:
        return FailingMetadataStore(
            base_path / "metadata.sqlite",
            event_log_path=base_path / "events.jsonl",
        )

    kernel, metadata_store, artifact_store, cache_store = _open_kernel(
        tmp_path,
        metadata_store_factory=build_metadata_store,
    )
    try:
        tx = kernel.begin_transaction(scope)
        staged = kernel.stage_memory_item(tx.tx_id, memory_factory(title="Atomic root commit"))

        with pytest.raises(RuntimeError, match="simulated lineage failure"):
            kernel.commit_transaction(tx.tx_id)

        assert metadata_store.get_memory_item(staged.memory_id) is None
        assert metadata_store.list_lineage_edges(tx_id=tx.tx_id) == ()
        assert metadata_store.get_transaction(tx.tx_id) is not None
        assert metadata_store.get_transaction(tx.tx_id).state is TxState.ACTIVE
        visible = {item.memory_id for item in kernel.list_visible_memory(tx.tx_id)}
        assert visible == {staged.memory_id}
    finally:
        _close_kernel(metadata_store, artifact_store, cache_store)


def test_root_commit_rolls_back_when_commit_event_persistence_fails(
    kernel,
    metadata_store,
    memory_factory,
    scope,
    monkeypatch,
) -> None:
    tx = kernel.begin_transaction(scope)
    staged = kernel.stage_memory_item(tx.tx_id, memory_factory(title="Event-backed root commit"))
    original_save_event = metadata_store.save_event

    def save_event(event) -> None:
        if event.event_type == "tx_commit":
            raise RuntimeError("simulated tx_commit event failure")
        original_save_event(event)

    monkeypatch.setattr(metadata_store, "save_event", save_event)

    with pytest.raises(RuntimeError, match="simulated tx_commit event failure"):
        kernel.commit_transaction(tx.tx_id)

    assert metadata_store.get_memory_item(staged.memory_id) is None
    assert metadata_store.get_transaction(tx.tx_id) is not None
    assert metadata_store.get_transaction(tx.tx_id).state is TxState.ACTIVE
    visible = {item.memory_id for item in kernel.list_visible_memory(tx.tx_id)}
    assert visible == {staged.memory_id}
    assert not any(event.event_type == "tx_commit" for event in metadata_store.list_events())


def test_child_commit_rolls_back_staged_move_when_transaction_update_fails(
    tmp_path: Path,
    memory_factory,
    scope,
) -> None:
    class FailingMetadataStore(SqliteMetadataStore):
        def save_transaction(self, transaction: TransactionRecord) -> None:
            if transaction.parent_tx_id is not None and transaction.state is TxState.COMMITTED:
                raise RuntimeError("simulated child commit failure")
            super().save_transaction(transaction)

    def build_metadata_store(base_path: Path) -> SqliteMetadataStore:
        return FailingMetadataStore(
            base_path / "metadata.sqlite",
            event_log_path=base_path / "events.jsonl",
        )

    kernel, metadata_store, artifact_store, cache_store = _open_kernel(
        tmp_path,
        metadata_store_factory=build_metadata_store,
    )
    try:
        root = kernel.begin_transaction(scope)
        child = kernel.begin_transaction(scope, parent_tx_id=root.tx_id)
        root_item = kernel.stage_memory_item(root.tx_id, memory_factory(title="Atomic parent"))
        child_item = kernel.stage_memory_item(child.tx_id, memory_factory(title="Atomic child"))

        with pytest.raises(RuntimeError, match="simulated child commit failure"):
            kernel.commit_transaction(child.tx_id)

        assert metadata_store.get_transaction(child.tx_id) is not None
        assert metadata_store.get_transaction(child.tx_id).state is TxState.ACTIVE
        visible_in_root = {item.memory_id for item in kernel.list_visible_memory(root.tx_id)}
        visible_in_child = {item.memory_id for item in kernel.list_visible_memory(child.tx_id)}
        assert visible_in_root == {root_item.memory_id}
        assert visible_in_child == {root_item.memory_id, child_item.memory_id}
        assert not any(
            event.event_type == "tx_commit_merged" for event in metadata_store.list_events()
        )
    finally:
        _close_kernel(metadata_store, artifact_store, cache_store)


def test_rollback_restores_staged_items_when_transaction_update_fails(
    tmp_path: Path,
    memory_factory,
    scope,
) -> None:
    class FailingMetadataStore(SqliteMetadataStore):
        def save_transaction(self, transaction: TransactionRecord) -> None:
            if transaction.state is TxState.ROLLED_BACK:
                raise RuntimeError("simulated rollback failure")
            super().save_transaction(transaction)

    def build_metadata_store(base_path: Path) -> SqliteMetadataStore:
        return FailingMetadataStore(
            base_path / "metadata.sqlite",
            event_log_path=base_path / "events.jsonl",
        )

    kernel, metadata_store, artifact_store, cache_store = _open_kernel(
        tmp_path,
        metadata_store_factory=build_metadata_store,
    )
    try:
        tx = kernel.begin_transaction(scope)
        staged = kernel.stage_memory_item(tx.tx_id, memory_factory(title="Atomic rollback"))

        with pytest.raises(RuntimeError, match="simulated rollback failure"):
            kernel.rollback_transaction(tx.tx_id)

        assert metadata_store.get_transaction(tx.tx_id) is not None
        assert metadata_store.get_transaction(tx.tx_id).state is TxState.ACTIVE
        visible = {item.memory_id for item in kernel.list_visible_memory(tx.tx_id)}
        assert visible == {staged.memory_id}
        assert not any(
            event.event_type == "tx_rollback" for event in metadata_store.list_events()
        )
    finally:
        _close_kernel(metadata_store, artifact_store, cache_store)
