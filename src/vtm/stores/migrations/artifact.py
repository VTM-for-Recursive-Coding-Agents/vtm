from __future__ import annotations

import sqlite3

from vtm.stores._sqlite_schema import has_table, list_columns

ARTIFACT_SCHEMA_VERSION = 2


def apply_artifact_migrations(conn: sqlite3.Connection, current_version: int) -> None:
    for version in range(current_version + 1, ARTIFACT_SCHEMA_VERSION + 1):
        if version == 1:
            _apply_schema_v1(conn)
            continue
        if version == 2:
            _apply_schema_v2(conn)
            continue
        raise ValueError(f"unsupported artifact schema migration target: {version}")


def _apply_schema_v1(conn: sqlite3.Connection) -> None:
    if has_table(conn, "artifacts"):
        rows = conn.execute(
            "SELECT artifact_id, sha256, relative_path, data FROM artifacts"
        ).fetchall()
        conn.execute("ALTER TABLE artifacts RENAME TO artifacts_legacy")
        conn.executescript(
            """
            CREATE TABLE artifacts (
                artifact_id TEXT PRIMARY KEY,
                sha256 TEXT NOT NULL,
                relative_path TEXT NOT NULL,
                data TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_artifacts_sha ON artifacts(sha256);
            """
        )
        conn.executemany(
            """
            INSERT INTO artifacts (artifact_id, sha256, relative_path, data)
            VALUES (?, ?, ?, ?)
            """,
            [
                (
                    row["artifact_id"],
                    row["sha256"],
                    row["relative_path"],
                    row["data"],
                )
                for row in rows
            ],
        )
        conn.execute("DROP TABLE artifacts_legacy")
        return

    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS artifacts (
            artifact_id TEXT PRIMARY KEY,
            sha256 TEXT NOT NULL,
            relative_path TEXT NOT NULL,
            data TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_artifacts_sha ON artifacts(sha256);
        """
    )


def _apply_schema_v2(conn: sqlite3.Connection) -> None:
    if "capture_state" not in list_columns(conn, "artifacts"):
        conn.execute(
            """
            ALTER TABLE artifacts
            ADD COLUMN capture_state TEXT NOT NULL DEFAULT 'committed'
            """
        )
    if "capture_group_id" not in list_columns(conn, "artifacts"):
        conn.execute(
            """
            ALTER TABLE artifacts
            ADD COLUMN capture_group_id TEXT
            """
        )
    if "committed_at" not in list_columns(conn, "artifacts"):
        conn.execute(
            """
            ALTER TABLE artifacts
            ADD COLUMN committed_at TEXT
            """
        )
    if "abandoned_at" not in list_columns(conn, "artifacts"):
        conn.execute(
            """
            ALTER TABLE artifacts
            ADD COLUMN abandoned_at TEXT
            """
        )
    if "actor" not in list_columns(conn, "artifacts"):
        conn.execute(
            """
            ALTER TABLE artifacts
            ADD COLUMN actor TEXT NOT NULL DEFAULT 'system'
            """
        )
    if "session_id" not in list_columns(conn, "artifacts"):
        conn.execute(
            """
            ALTER TABLE artifacts
            ADD COLUMN session_id TEXT
            """
        )

    conn.executescript(
        """
        CREATE INDEX IF NOT EXISTS idx_artifacts_state ON artifacts(capture_state);
        CREATE INDEX IF NOT EXISTS idx_artifacts_group ON artifacts(capture_group_id);
        """
    )
