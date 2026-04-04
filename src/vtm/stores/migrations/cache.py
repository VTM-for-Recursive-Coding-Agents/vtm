from __future__ import annotations

import sqlite3

CACHE_SCHEMA_VERSION = 1


def apply_cache_migrations(conn: sqlite3.Connection, current_version: int) -> None:
    for version in range(current_version + 1, CACHE_SCHEMA_VERSION + 1):
        if version == 1:
            _apply_schema_v1(conn)
            continue
        raise ValueError(f"unsupported cache schema migration target: {version}")


def _apply_schema_v1(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS cache_entries (
            digest TEXT PRIMARY KEY,
            tool_name TEXT NOT NULL,
            created_at TEXT NOT NULL,
            expires_at TEXT,
            data TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_cache_entries_tool_name ON cache_entries(tool_name);
        """
    )
