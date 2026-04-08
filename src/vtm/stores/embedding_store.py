"""SQLite-backed derived embedding index store."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from pathlib import Path

from vtm.embeddings import EmbeddingIndexEntry
from vtm.stores._sqlite_schema import (
    ensure_schema_tracking_tables,
    read_schema_version,
    record_schema_migration,
)
from vtm.stores.migrations.embedding import (
    EMBEDDING_SCHEMA_VERSION,
    apply_embedding_migrations,
)


class SqliteEmbeddingIndexStore:
    """Persists derived embeddings keyed by memory and adapter id."""

    def __init__(self, db_path: str | Path) -> None:
        """Open or initialize the SQLite embedding index database."""
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()

    def _init_schema(self) -> None:
        ensure_schema_tracking_tables(self._conn)
        current_version = read_schema_version(self._conn)
        if current_version > EMBEDDING_SCHEMA_VERSION:
            raise ValueError(
                f"embedding store schema version {current_version} is newer than supported "
                f"{EMBEDDING_SCHEMA_VERSION}"
            )
        with self._conn:
            apply_embedding_migrations(self._conn, current_version)
            for version in range(current_version + 1, EMBEDDING_SCHEMA_VERSION + 1):
                record_schema_migration(self._conn, version)

    def save_entry(self, entry: EmbeddingIndexEntry) -> None:
        self._conn.execute(
            """
            INSERT INTO embedding_index_entries (
                memory_id, adapter_id, content_digest, vector_json, created_at, updated_at, data
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(memory_id, adapter_id) DO UPDATE SET
                content_digest = excluded.content_digest,
                vector_json = excluded.vector_json,
                created_at = excluded.created_at,
                updated_at = excluded.updated_at,
                data = excluded.data
            """,
            (
                entry.memory_id,
                entry.adapter_id,
                entry.content_digest,
                json.dumps(entry.vector),
                entry.created_at.isoformat(),
                entry.updated_at.isoformat(),
                entry.to_json(),
            ),
        )
        self._conn.commit()

    def get_entry(self, memory_id: str, adapter_id: str) -> EmbeddingIndexEntry | None:
        row = self._conn.execute(
            """
            SELECT data
            FROM embedding_index_entries
            WHERE memory_id = ? AND adapter_id = ?
            """,
            (memory_id, adapter_id),
        ).fetchone()
        if row is None:
            return None
        return EmbeddingIndexEntry.from_json(row["data"])

    def list_entries(
        self,
        *,
        adapter_id: str | None = None,
    ) -> Sequence[EmbeddingIndexEntry]:
        sql = "SELECT data FROM embedding_index_entries"
        params: list[str] = []
        if adapter_id is not None:
            sql += " WHERE adapter_id = ?"
            params.append(adapter_id)
        sql += " ORDER BY memory_id ASC, adapter_id ASC"
        rows = self._conn.execute(sql, params).fetchall()
        return tuple(EmbeddingIndexEntry.from_json(row["data"]) for row in rows)

    def delete_entry(self, memory_id: str, adapter_id: str) -> None:
        self._conn.execute(
            """
            DELETE FROM embedding_index_entries
            WHERE memory_id = ? AND adapter_id = ?
            """,
            (memory_id, adapter_id),
        )
        self._conn.commit()

    def __enter__(self) -> SqliteEmbeddingIndexStore:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
