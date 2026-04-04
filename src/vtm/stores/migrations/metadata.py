from __future__ import annotations

import sqlite3

from vtm.stores._sqlite_schema import has_table, list_columns

METADATA_SCHEMA_VERSION = 2


def apply_metadata_migrations(conn: sqlite3.Connection, current_version: int) -> None:
    for version in range(current_version + 1, METADATA_SCHEMA_VERSION + 1):
        if version == 1:
            _apply_schema_v1(conn)
            continue
        if version == 2:
            _apply_schema_v2(conn)
            continue
        raise ValueError(f"unsupported metadata schema migration target: {version}")


def _apply_schema_v1(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS memory_items (
            memory_id TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            status TEXT NOT NULL,
            scope_kind TEXT NOT NULL,
            scope_id TEXT NOT NULL,
            tx_id TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            data TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_memory_items_status ON memory_items(status);
        CREATE INDEX IF NOT EXISTS idx_memory_items_scope ON memory_items(scope_kind, scope_id);
        CREATE INDEX IF NOT EXISTS idx_memory_items_tx ON memory_items(tx_id);

        CREATE TABLE IF NOT EXISTS lineage_edges (
            parent_id TEXT NOT NULL,
            child_id TEXT NOT NULL,
            edge_type TEXT NOT NULL,
            tx_id TEXT,
            created_at TEXT NOT NULL,
            data TEXT NOT NULL,
            PRIMARY KEY (parent_id, child_id, edge_type, created_at)
        );
        CREATE INDEX IF NOT EXISTS idx_lineage_edges_child ON lineage_edges(child_id);
        CREATE INDEX IF NOT EXISTS idx_lineage_edges_tx ON lineage_edges(tx_id);

        CREATE TABLE IF NOT EXISTS transactions (
            tx_id TEXT PRIMARY KEY,
            parent_tx_id TEXT,
            state TEXT NOT NULL,
            scope_kind TEXT NOT NULL,
            scope_id TEXT NOT NULL,
            opened_at TEXT NOT NULL,
            committed_at TEXT,
            rolled_back_at TEXT,
            data TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_transactions_parent ON transactions(parent_tx_id);
        CREATE INDEX IF NOT EXISTS idx_transactions_state ON transactions(state);

        CREATE TABLE IF NOT EXISTS staged_memory_items (
            tx_id TEXT NOT NULL,
            stage_order INTEGER NOT NULL,
            memory_id TEXT NOT NULL,
            data TEXT NOT NULL,
            PRIMARY KEY (tx_id, stage_order)
        );
        CREATE INDEX IF NOT EXISTS idx_staged_memory_items_tx_order
            ON staged_memory_items(tx_id, stage_order);
        """
    )

    if not has_table(conn, "events"):
        conn.executescript(
            """
            CREATE TABLE events (
                event_id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                occurred_at TEXT NOT NULL,
                tx_id TEXT,
                memory_id TEXT,
                cache_digest TEXT,
                exported_to_jsonl INTEGER NOT NULL DEFAULT 0,
                data TEXT NOT NULL
            );
            """
        )
    elif "exported_to_jsonl" not in list_columns(conn, "events"):
        conn.execute(
            """
            ALTER TABLE events
            ADD COLUMN exported_to_jsonl INTEGER NOT NULL DEFAULT 0
            """
        )

    conn.executescript(
        """
        CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
        CREATE INDEX IF NOT EXISTS idx_events_tx ON events(tx_id);
        CREATE INDEX IF NOT EXISTS idx_events_memory ON events(memory_id);
        CREATE INDEX IF NOT EXISTS idx_events_export
            ON events(exported_to_jsonl, occurred_at, event_id);
        """
    )


def _apply_schema_v2(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS event_export_state (
            export_name TEXT PRIMARY KEY,
            last_exported_event_id TEXT,
            last_exported_occurred_at TEXT,
            last_exported_at TEXT,
            full_rebuild_count INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL
        );
        """
    )
