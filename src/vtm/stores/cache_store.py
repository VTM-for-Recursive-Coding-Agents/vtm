"""SQLite-backed cache store with optional event logging."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from vtm.base import utc_now
from vtm.cache import CacheEntry, CacheKey
from vtm.enums import FreshnessMode
from vtm.events import MemoryEvent
from vtm.stores._sqlite_schema import (
    ensure_schema_tracking_tables,
    read_schema_version,
    record_schema_migration,
)
from vtm.stores.base import EventStore
from vtm.stores.migrations.cache import CACHE_SCHEMA_VERSION, apply_cache_migrations


class SqliteCacheStore:
    """Persists cache entries and optionally emits cache hit/miss events."""

    def __init__(self, db_path: str | Path, *, event_store: EventStore | None = None) -> None:
        """Open or initialize the SQLite cache database."""
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._event_store = event_store
        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()

    def _init_schema(self) -> None:
        ensure_schema_tracking_tables(self._conn)
        current_version = read_schema_version(self._conn)
        if current_version > CACHE_SCHEMA_VERSION:
            raise ValueError(
                f"cache store schema version {current_version} is newer than supported "
                f"{CACHE_SCHEMA_VERSION}"
            )
        with self._conn:
            apply_cache_migrations(self._conn, current_version)
            for version in range(current_version + 1, CACHE_SCHEMA_VERSION + 1):
                record_schema_migration(self._conn, version)

    def save_cache_entry(self, entry: CacheEntry) -> None:
        self._conn.execute(
            """
            INSERT INTO cache_entries (digest, tool_name, created_at, expires_at, data)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(digest) DO UPDATE SET
                tool_name = excluded.tool_name,
                created_at = excluded.created_at,
                expires_at = excluded.expires_at,
                data = excluded.data
            """,
            (
                entry.key.digest,
                entry.key.tool_name,
                entry.created_at.isoformat(),
                entry.expires_at.isoformat() if entry.expires_at else None,
                entry.to_json(),
            ),
        )
        self._conn.commit()

    def get_cache_entry(self, key: CacheKey, *, now: datetime | None = None) -> CacheEntry | None:
        row = self._conn.execute(
            "SELECT data FROM cache_entries WHERE digest = ?",
            (key.digest,),
        ).fetchone()
        current_time = self._normalize_timestamp(now)
        if row is None:
            self._log_cache_event("cache_miss", key)
            return None

        entry = CacheEntry.from_json(row["data"])
        expires_at = self._normalize_timestamp(entry.expires_at) if entry.expires_at else None
        if expires_at is not None and current_time > expires_at:
            if entry.freshness_mode is not FreshnessMode.ALLOW_STALE:
                self._log_cache_event("cache_miss", key)
                return None

        updated_entry = entry.model_copy(update={"hit_count": entry.hit_count + 1})
        self.save_cache_entry(updated_entry)
        self._log_cache_event("cache_hit", key)
        return updated_entry

    def delete_cache_entry(self, key: CacheKey) -> None:
        self._conn.execute("DELETE FROM cache_entries WHERE digest = ?", (key.digest,))
        self._conn.commit()

    def list_cache_entries(self) -> tuple[CacheEntry, ...]:
        rows = self._conn.execute(
            "SELECT data FROM cache_entries ORDER BY created_at ASC, digest ASC"
        ).fetchall()
        return tuple(CacheEntry.from_json(row["data"]) for row in rows)

    def _log_cache_event(self, event_type: str, key: CacheKey) -> None:
        if self._event_store is None:
            return
        self._event_store.save_event(
            MemoryEvent(
                event_type=event_type,
                cache_digest=key.digest,
                tool_name=key.tool_name,
                payload={"tool_name": key.tool_name},
            )
        )

    def __enter__(self) -> SqliteCacheStore:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _normalize_timestamp(self, value: datetime | None) -> datetime:
        if value is None:
            return utc_now()
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)
