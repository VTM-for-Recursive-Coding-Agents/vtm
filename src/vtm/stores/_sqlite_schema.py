from __future__ import annotations

import sqlite3

from vtm.base import utc_now


def ensure_schema_tracking_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS schema_meta (
            singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
            schema_version INTEGER NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            applied_at TEXT NOT NULL
        );
        """
    )


def read_schema_version(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        "SELECT schema_version FROM schema_meta WHERE singleton = 1",
    ).fetchone()
    if row is None:
        return 0
    return int(row["schema_version"])


def record_schema_migration(conn: sqlite3.Connection, version: int) -> None:
    applied_at = utc_now().isoformat()
    conn.execute(
        """
        INSERT INTO schema_meta (singleton, schema_version, updated_at)
        VALUES (1, ?, ?)
        ON CONFLICT(singleton) DO UPDATE SET
            schema_version = excluded.schema_version,
            updated_at = excluded.updated_at
        """,
        (version, applied_at),
    )
    conn.execute(
        """
        INSERT OR REPLACE INTO schema_migrations (version, applied_at)
        VALUES (?, ?)
        """,
        (version, applied_at),
    )


def has_table(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        """
        SELECT 1
        FROM sqlite_master
        WHERE type = 'table' AND name = ?
        """,
        (table_name,),
    ).fetchone()
    return row is not None


def list_columns(conn: sqlite3.Connection, table_name: str) -> tuple[str, ...]:
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return tuple(str(row["name"]) for row in rows)
