"""SQLite-backed metadata and canonical event storage."""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TextIO, TypeVar

from vtm.base import utc_now
from vtm.enums import ValidityStatus
from vtm.events import MemoryEvent
from vtm.memory_items import LineageEdge, MemoryItem, VisibilityScope
from vtm.stores._sqlite_schema import (
    ensure_schema_tracking_tables,
    read_schema_version,
    record_schema_migration,
)
from vtm.stores.migrations.metadata import (
    METADATA_SCHEMA_VERSION,
    apply_metadata_migrations,
)
from vtm.transactions import TransactionRecord

ResultT = TypeVar("ResultT")
type ExportStateRow = tuple[str, str | None, str | None, str, str]
JSONL_EXPORT_NAME = "jsonl"


class SqliteMetadataStore:
    """SQLite implementation of metadata storage and the canonical event ledger."""

    def __init__(self, db_path: str | Path, *, event_log_path: str | Path | None = None) -> None:
        """Open or initialize the metadata store and optional JSONL export file."""
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._event_log_path = Path(event_log_path) if event_log_path is not None else None
        if self._event_log_path is not None:
            self._event_log_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        self._transaction_depth = 0
        self._init_schema()

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()

    def _init_schema(self) -> None:
        ensure_schema_tracking_tables(self._conn)
        current_version = read_schema_version(self._conn)
        if current_version > METADATA_SCHEMA_VERSION:
            raise ValueError(
                f"metadata store schema version {current_version} is newer than supported "
                f"{METADATA_SCHEMA_VERSION}"
            )
        with self._conn:
            apply_metadata_migrations(self._conn, current_version)
            for version in range(current_version + 1, METADATA_SCHEMA_VERSION + 1):
                record_schema_migration(self._conn, version)

    def save_memory_item(self, item: MemoryItem) -> None:
        self._conn.execute(
            """
            INSERT INTO memory_items (
                memory_id, kind, status, scope_kind, scope_id, tx_id, created_at, updated_at, data
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(memory_id) DO UPDATE SET
                kind = excluded.kind,
                status = excluded.status,
                scope_kind = excluded.scope_kind,
                scope_id = excluded.scope_id,
                tx_id = excluded.tx_id,
                created_at = excluded.created_at,
                updated_at = excluded.updated_at,
                data = excluded.data
            """,
            (
                item.memory_id,
                item.kind.value,
                item.validity.status.value,
                item.visibility.kind.value,
                item.visibility.scope_id,
                item.tx_id,
                item.created_at.isoformat(),
                item.updated_at.isoformat(),
                item.to_json(),
            ),
        )
        self._maybe_commit()

    def get_memory_item(self, memory_id: str) -> MemoryItem | None:
        row = self._conn.execute(
            "SELECT data FROM memory_items WHERE memory_id = ?",
            (memory_id,),
        ).fetchone()
        if row is None:
            return None
        return MemoryItem.from_json(row["data"])

    def list_memory_items(self) -> Sequence[MemoryItem]:
        rows = self._conn.execute(
            "SELECT data FROM memory_items ORDER BY created_at ASC, memory_id ASC"
        ).fetchall()
        return tuple(MemoryItem.from_json(row["data"]) for row in rows)

    def query_memory_items(
        self,
        scopes: Sequence[VisibilityScope],
        statuses: Sequence[ValidityStatus] | None = None,
        allow_quarantined: bool = False,
    ) -> Sequence[MemoryItem]:
        sql = "SELECT data FROM memory_items"
        clauses: list[str] = []
        params: list[str] = []

        if scopes:
            scope_clause = " OR ".join("(scope_kind = ? AND scope_id = ?)" for _ in scopes)
            clauses.append(f"({scope_clause})")
            for scope in scopes:
                params.extend([scope.kind.value, scope.scope_id])

        if statuses:
            clauses.append(f"status IN ({','.join('?' for _ in statuses)})")
            params.extend(status.value for status in statuses)

        if not allow_quarantined:
            clauses.append("status != ?")
            params.append(ValidityStatus.QUARANTINED.value)

        if clauses:
            sql = f"{sql} WHERE {' AND '.join(clauses)}"

        sql = f"{sql} ORDER BY updated_at DESC, memory_id ASC"
        rows = self._conn.execute(sql, params).fetchall()
        return tuple(MemoryItem.from_json(row["data"]) for row in rows)

    def save_lineage_edge(self, edge: LineageEdge) -> None:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO lineage_edges (
                parent_id, child_id, edge_type, tx_id, created_at, data
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                edge.parent_id,
                edge.child_id,
                edge.edge_type,
                edge.tx_id,
                edge.created_at.isoformat(),
                edge.to_json(),
            ),
        )
        self._maybe_commit()

    def list_lineage_edges(
        self,
        *,
        child_id: str | None = None,
        tx_id: str | None = None,
    ) -> Sequence[LineageEdge]:
        sql = "SELECT data FROM lineage_edges"
        clauses: list[str] = []
        params: list[str] = []
        if child_id is not None:
            clauses.append("child_id = ?")
            params.append(child_id)
        if tx_id is not None:
            clauses.append("tx_id = ?")
            params.append(tx_id)
        if clauses:
            sql = f"{sql} WHERE {' AND '.join(clauses)}"
        sql = f"{sql} ORDER BY created_at ASC"
        rows = self._conn.execute(sql, params).fetchall()
        return tuple(LineageEdge.from_json(row["data"]) for row in rows)

    def save_transaction(self, transaction: TransactionRecord) -> None:
        self._conn.execute(
            """
            INSERT INTO transactions (
                tx_id, parent_tx_id, state, scope_kind, scope_id, opened_at, committed_at,
                rolled_back_at, data
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(tx_id) DO UPDATE SET
                parent_tx_id = excluded.parent_tx_id,
                state = excluded.state,
                scope_kind = excluded.scope_kind,
                scope_id = excluded.scope_id,
                opened_at = excluded.opened_at,
                committed_at = excluded.committed_at,
                rolled_back_at = excluded.rolled_back_at,
                data = excluded.data
            """,
            (
                transaction.tx_id,
                transaction.parent_tx_id,
                transaction.state.value,
                transaction.visibility.kind.value,
                transaction.visibility.scope_id,
                transaction.opened_at.isoformat(),
                transaction.committed_at.isoformat() if transaction.committed_at else None,
                transaction.rolled_back_at.isoformat() if transaction.rolled_back_at else None,
                transaction.to_json(),
            ),
        )
        self._maybe_commit()

    def get_transaction(self, tx_id: str) -> TransactionRecord | None:
        row = self._conn.execute(
            "SELECT data FROM transactions WHERE tx_id = ?",
            (tx_id,),
        ).fetchone()
        if row is None:
            return None
        return TransactionRecord.from_json(row["data"])

    def list_transactions(self) -> Sequence[TransactionRecord]:
        rows = self._conn.execute(
            "SELECT data FROM transactions ORDER BY opened_at ASC, tx_id ASC"
        ).fetchall()
        return tuple(TransactionRecord.from_json(row["data"]) for row in rows)

    def append_staged_memory_item(self, tx_id: str, item: MemoryItem) -> None:
        next_stage_order = self._next_stage_order(tx_id)
        self._conn.execute(
            """
            INSERT INTO staged_memory_items (tx_id, stage_order, memory_id, data)
            VALUES (?, ?, ?, ?)
            """,
            (tx_id, next_stage_order, item.memory_id, item.to_json()),
        )
        self._maybe_commit()

    def list_staged_memory_items(self, tx_id: str) -> Sequence[MemoryItem]:
        rows = self._conn.execute(
            """
            SELECT data
            FROM staged_memory_items
            WHERE tx_id = ?
            ORDER BY stage_order ASC
            """,
            (tx_id,),
        ).fetchall()
        return tuple(MemoryItem.from_json(row["data"]) for row in rows)

    def move_staged_memory_items(self, source_tx_id: str, target_tx_id: str) -> None:
        source_rows = self._conn.execute(
            """
            SELECT memory_id, data
            FROM staged_memory_items
            WHERE tx_id = ?
            ORDER BY stage_order ASC
            """,
            (source_tx_id,),
        ).fetchall()
        if not source_rows:
            return

        try:
            next_stage_order = self._next_stage_order(target_tx_id)
            for offset, row in enumerate(source_rows):
                self._conn.execute(
                    """
                    INSERT INTO staged_memory_items (tx_id, stage_order, memory_id, data)
                    VALUES (?, ?, ?, ?)
                    """,
                    (target_tx_id, next_stage_order + offset, row["memory_id"], row["data"]),
                )
            self._conn.execute(
                "DELETE FROM staged_memory_items WHERE tx_id = ?",
                (source_tx_id,),
            )
            self._maybe_commit()
        except Exception:
            if self._transaction_depth == 0:
                self._conn.rollback()
            raise

    def clear_staged_memory_items(self, tx_id: str) -> None:
        self._conn.execute("DELETE FROM staged_memory_items WHERE tx_id = ?", (tx_id,))
        self._maybe_commit()

    def run_atomically(self, operation: Callable[[], ResultT]) -> ResultT:
        if self._transaction_depth > 0:
            self._transaction_depth += 1
            try:
                return operation()
            finally:
                self._transaction_depth -= 1
        with self._conn:
            self._transaction_depth += 1
            try:
                return operation()
            finally:
                self._transaction_depth -= 1

    def save_event(self, event: MemoryEvent) -> None:
        self._conn.execute(
            """
            INSERT INTO events (
                event_id, event_type, occurred_at, tx_id, memory_id, cache_digest,
                exported_to_jsonl, data
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(event_id) DO UPDATE SET
                event_type = excluded.event_type,
                occurred_at = excluded.occurred_at,
                tx_id = excluded.tx_id,
                memory_id = excluded.memory_id,
                cache_digest = excluded.cache_digest,
                exported_to_jsonl = excluded.exported_to_jsonl,
                data = excluded.data
            """,
            (
                event.event_id,
                event.event_type,
                event.occurred_at.isoformat(),
                event.tx_id,
                event.memory_id,
                event.cache_digest,
                0,
                event.to_json(),
            ),
        )
        self._maybe_commit()

    def export_events_to_jsonl(self) -> int:
        if self._event_log_path is None:
            raise RuntimeError("no event_log_path configured for JSONL export")

        self._reconcile_jsonl_progress_from_file()
        rows = self._conn.execute(
            """
            SELECT event_id, occurred_at, data
            FROM events
            WHERE exported_to_jsonl = 0
            ORDER BY occurred_at ASC, event_id ASC
            """
        ).fetchall()
        if not rows:
            return 0

        exported = 0
        with self._event_log_path.open("a", encoding="utf-8") as handle:
            for row in rows:
                payload = row["data"]
                event = MemoryEvent.from_json(payload)
                self._append_jsonl_event(handle, payload)
                self._mark_event_exported(event)
                exported += 1
        return exported

    def rebuild_events_jsonl(self) -> int:
        if self._event_log_path is None:
            raise RuntimeError("no event_log_path configured for JSONL export")

        rows = self._conn.execute(
            """
            SELECT event_id, occurred_at, data
            FROM events
            ORDER BY occurred_at ASC, event_id ASC
            """
        ).fetchall()
        with self._event_log_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                self._append_jsonl_event(handle, row["data"])

        with self._conn:
            self._conn.execute("UPDATE events SET exported_to_jsonl = 1")
            self._conn.execute(
                """
                INSERT INTO event_export_state (
                    export_name,
                    last_exported_event_id,
                    last_exported_occurred_at,
                    last_exported_at,
                    full_rebuild_count,
                    updated_at
                ) VALUES (?, ?, ?, ?, 1, ?)
                ON CONFLICT(export_name) DO UPDATE SET
                    last_exported_event_id = excluded.last_exported_event_id,
                    last_exported_occurred_at = excluded.last_exported_occurred_at,
                    last_exported_at = excluded.last_exported_at,
                    full_rebuild_count = event_export_state.full_rebuild_count + 1,
                    updated_at = excluded.updated_at
                """,
                self._export_state_values(rows[-1] if rows else None),
            )
        return len(rows)

    def get_event_export_state(self) -> dict[str, object] | None:
        row = self._conn.execute(
            """
            SELECT
                export_name,
                last_exported_event_id,
                last_exported_occurred_at,
                last_exported_at,
                full_rebuild_count,
                updated_at
            FROM event_export_state
            WHERE export_name = ?
            """,
            (JSONL_EXPORT_NAME,),
        ).fetchone()
        if row is None:
            return None
        return {
            "export_name": row["export_name"],
            "last_exported_event_id": row["last_exported_event_id"],
            "last_exported_occurred_at": row["last_exported_occurred_at"],
            "last_exported_at": row["last_exported_at"],
            "full_rebuild_count": row["full_rebuild_count"],
            "updated_at": row["updated_at"],
        }

    def get_event(self, event_id: str) -> MemoryEvent | None:
        row = self._conn.execute(
            "SELECT data FROM events WHERE event_id = ?",
            (event_id,),
        ).fetchone()
        if row is None:
            return None
        return MemoryEvent.from_json(row["data"])

    def list_events(self) -> Sequence[MemoryEvent]:
        rows = self._conn.execute(
            "SELECT data FROM events ORDER BY occurred_at ASC, event_id ASC"
        ).fetchall()
        return tuple(MemoryEvent.from_json(row["data"]) for row in rows)

    def __enter__(self) -> SqliteMetadataStore:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _append_jsonl_event(self, handle: TextIO, payload: str) -> None:
        handle.write(payload)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())

    def _reconcile_jsonl_progress_from_file(self) -> None:
        recovered_events = self._recover_jsonl_events_from_file()
        if not recovered_events:
            return

        recovered_event_ids: list[str] = []
        seen_event_ids: set[str] = set()
        for event in recovered_events:
            if event.event_id in seen_event_ids:
                continue
            if self.get_event(event.event_id) != event:
                continue
            recovered_event_ids.append(event.event_id)
            seen_event_ids.add(event.event_id)

        if not recovered_event_ids:
            return

        rows = self._conn.execute(
            """
            SELECT event_id, occurred_at, exported_to_jsonl
            FROM events
            ORDER BY occurred_at ASC, event_id ASC
            """
        ).fetchall()
        recovered_event_id_set = set(recovered_event_ids)

        contiguous_prefix: list[sqlite3.Row] = []
        for row in rows:
            if row["event_id"] not in recovered_event_id_set:
                break
            contiguous_prefix.append(row)

        unmarked_prefix_ids = [
            row["event_id"] for row in contiguous_prefix if row["exported_to_jsonl"] == 0
        ]
        if not unmarked_prefix_ids:
            return

        last_row = contiguous_prefix[-1]
        now = utc_now().isoformat()
        with self._conn:
            placeholders = ",".join("?" for _ in unmarked_prefix_ids)
            self._conn.execute(
                f"UPDATE events SET exported_to_jsonl = 1 WHERE event_id IN ({placeholders})",
                unmarked_prefix_ids,
            )
            self._conn.execute(
                """
                INSERT INTO event_export_state (
                    export_name,
                    last_exported_event_id,
                    last_exported_occurred_at,
                    last_exported_at,
                    full_rebuild_count,
                    updated_at
                ) VALUES (?, ?, ?, ?, 0, ?)
                ON CONFLICT(export_name) DO UPDATE SET
                    last_exported_event_id = excluded.last_exported_event_id,
                    last_exported_occurred_at = excluded.last_exported_occurred_at,
                    last_exported_at = excluded.last_exported_at,
                    updated_at = excluded.updated_at
                """,
                (
                    JSONL_EXPORT_NAME,
                    last_row["event_id"],
                    last_row["occurred_at"],
                    now,
                    now,
                ),
            )

    def _recover_jsonl_events_from_file(self) -> tuple[MemoryEvent, ...]:
        if self._event_log_path is None or not self._event_log_path.exists():
            return ()

        payload = self._event_log_path.read_bytes()
        if not payload:
            return ()

        recovered: list[MemoryEvent] = []
        safe_offset = 0
        offset = 0
        for raw_line in payload.splitlines(keepends=True):
            offset += len(raw_line)
            is_complete_line = raw_line.endswith(b"\n")
            line = raw_line[:-1] if is_complete_line else raw_line
            if not line:
                break
            try:
                event = MemoryEvent.from_json(line.decode("utf-8"))
            except Exception:
                break
            recovered.append(event)
            safe_offset = offset
            if not is_complete_line:
                break

        if safe_offset < len(payload):
            self._event_log_path.write_bytes(payload[:safe_offset])
        return tuple(recovered)

    def _mark_event_exported(self, event: MemoryEvent) -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE events SET exported_to_jsonl = 1 WHERE event_id = ?",
                (event.event_id,),
            )
            self._conn.execute(
                """
                INSERT INTO event_export_state (
                    export_name,
                    last_exported_event_id,
                    last_exported_occurred_at,
                    last_exported_at,
                    full_rebuild_count,
                    updated_at
                ) VALUES (?, ?, ?, ?, 0, ?)
                ON CONFLICT(export_name) DO UPDATE SET
                    last_exported_event_id = excluded.last_exported_event_id,
                    last_exported_occurred_at = excluded.last_exported_occurred_at,
                    last_exported_at = excluded.last_exported_at,
                    updated_at = excluded.updated_at
                """,
                (
                    JSONL_EXPORT_NAME,
                    event.event_id,
                    event.occurred_at.isoformat(),
                    utc_now().isoformat(),
                    utc_now().isoformat(),
                ),
            )

    def _export_state_values(self, row: sqlite3.Row | None) -> ExportStateRow:
        now = utc_now().isoformat()
        if row is None:
            return (JSONL_EXPORT_NAME, None, None, now, now)
        return (
            JSONL_EXPORT_NAME,
            row["event_id"],
            row["occurred_at"],
            now,
            now,
        )

    def _next_stage_order(self, tx_id: str) -> int:
        row = self._conn.execute(
            """
            SELECT COALESCE(MAX(stage_order) + 1, 0) AS next_stage_order
            FROM staged_memory_items
            WHERE tx_id = ?
            """,
            (tx_id,),
        ).fetchone()
        return int(row["next_stage_order"])

    def _maybe_commit(self) -> None:
        if self._transaction_depth == 0:
            self._conn.commit()
