"""Embedding-index schema migrations."""

from __future__ import annotations

import sqlite3

EMBEDDING_SCHEMA_VERSION = 1


def apply_embedding_migrations(conn: sqlite3.Connection, current_version: int) -> None:
    """Apply all pending embedding-index migrations in order."""
    for version in range(current_version + 1, EMBEDDING_SCHEMA_VERSION + 1):
        if version == 1:
            _apply_schema_v1(conn)
            continue
        raise ValueError(f"unsupported embedding schema migration target: {version}")


def _apply_schema_v1(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS embedding_index_entries (
            memory_id TEXT NOT NULL,
            adapter_id TEXT NOT NULL,
            content_digest TEXT NOT NULL,
            vector_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            data TEXT NOT NULL,
            PRIMARY KEY (memory_id, adapter_id)
        );
        CREATE INDEX IF NOT EXISTS idx_embedding_index_entries_adapter_id
            ON embedding_index_entries(adapter_id);
        CREATE INDEX IF NOT EXISTS idx_embedding_index_entries_content_digest
            ON embedding_index_entries(content_digest);
        """
    )
