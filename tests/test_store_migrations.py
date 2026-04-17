from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import pytest

from vtm.artifacts import ArtifactRecord
from vtm.cache import CacheEntry, CacheKey
from vtm.events import MemoryEvent
from vtm.fingerprints import EnvFingerprint, RepoFingerprint, ToolVersion
from vtm.stores.artifact_store import FilesystemArtifactStore
from vtm.stores.cache_store import SqliteCacheStore
from vtm.stores.sqlite_store import SqliteMetadataStore

FIXTURE_ROOT = Path("tests/fixtures/migrations")


def _cache_key() -> CacheKey:
    return CacheKey.from_parts(
        "parser",
        {"mode": "fast"},
        RepoFingerprint(
            repo_root="/tmp/repo",
            branch="main",
            head_commit="abc123",
            tree_digest="tree-1",
            dirty_digest="dirty-1",
        ),
        EnvFingerprint(
            python_version="3.12.8",
            platform="darwin-arm64",
            tool_versions=(ToolVersion(name="pytest", version="8.3.4"),),
        ),
    )


def _read_schema_version(db_path: Path) -> int:
    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT schema_version FROM schema_meta WHERE singleton = 1").fetchone()
    return int(row[0])


def _load_sql_fixture(db_path: Path, relative_path: str) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    fixture_path = FIXTURE_ROOT / relative_path
    with sqlite3.connect(db_path) as conn:
        conn.executescript(fixture_path.read_text(encoding="utf-8"))


def _write_artifact_blob(root: Path, relative_path: str, payload: bytes) -> None:
    blob_path = root / relative_path
    blob_path.parent.mkdir(parents=True, exist_ok=True)
    blob_path.write_bytes(payload)


def _write_future_schema_version(db_path: Path, version: int) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE schema_meta (
                singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                schema_version INTEGER NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE schema_migrations (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL
            );
            """
        )
        conn.execute(
            """
            INSERT INTO schema_meta (singleton, schema_version, updated_at)
            VALUES (1, ?, '2026-04-03T00:00:00+00:00')
            """,
            (version,),
        )


def test_metadata_store_upgrades_legacy_events_schema(tmp_path: Path) -> None:
    db_path = tmp_path / "metadata.sqlite"
    legacy_event = MemoryEvent(event_id="evt_legacy", event_type="legacy")
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE events (
                event_id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                occurred_at TEXT NOT NULL,
                tx_id TEXT,
                memory_id TEXT,
                cache_digest TEXT,
                data TEXT NOT NULL
            );
            """
        )
        conn.execute(
            """
            INSERT INTO events (
                event_id, event_type, occurred_at, tx_id, memory_id, cache_digest, data
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                legacy_event.event_id,
                legacy_event.event_type,
                legacy_event.occurred_at.isoformat(),
                legacy_event.tx_id,
                legacy_event.memory_id,
                legacy_event.cache_digest,
                legacy_event.to_json(),
            ),
        )

    store = SqliteMetadataStore(db_path, event_log_path=tmp_path / "events.jsonl")
    try:
        assert store.get_event(legacy_event.event_id) == legacy_event
        assert _read_schema_version(db_path) == 2
        with sqlite3.connect(db_path) as conn:
            columns = tuple(row[1] for row in conn.execute("PRAGMA table_info(events)").fetchall())
            export_state_tables = conn.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table' AND name = 'event_export_state'
                """
            ).fetchall()
        assert "exported_to_jsonl" in columns
        assert export_state_tables == [("event_export_state",)]
    finally:
        store.close()


def test_cache_store_upgrades_legacy_schema_without_losing_entries(tmp_path: Path) -> None:
    db_path = tmp_path / "cache.sqlite"
    entry = CacheEntry(key=_cache_key(), value={"answer": 42})
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE cache_entries (
                digest TEXT PRIMARY KEY,
                tool_name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT,
                data TEXT NOT NULL
            );
            CREATE INDEX idx_cache_entries_tool_name ON cache_entries(tool_name);
            """
        )
        conn.execute(
            """
            INSERT INTO cache_entries (digest, tool_name, created_at, expires_at, data)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                entry.key.digest,
                entry.key.tool_name,
                entry.created_at.isoformat(),
                entry.expires_at.isoformat() if entry.expires_at else None,
                entry.to_json(),
            ),
        )

    store = SqliteCacheStore(db_path)
    try:
        assert store.list_cache_entries() == (entry,)
        assert _read_schema_version(db_path) == 1
    finally:
        store.close()


def test_artifact_store_upgrades_legacy_unique_sha_schema_and_preserves_provenance(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifacts"
    blob_root = root / "sha256"
    blob_root.mkdir(parents=True)
    index_path = root / "artifact_index.sqlite"
    payload = b"legacy artifact"
    digest = hashlib.sha256(payload).hexdigest()
    (blob_root / digest).write_bytes(payload)
    legacy_record = ArtifactRecord(
        artifact_id="art_legacy",
        sha256=digest,
        relative_path=f"sha256/{digest}",
        size_bytes=len(payload),
        content_type="text/plain",
        tool_name="legacy-tool",
        metadata={"capture": "legacy"},
    )

    with sqlite3.connect(index_path) as conn:
        conn.executescript(
            """
            CREATE TABLE artifacts (
                artifact_id TEXT PRIMARY KEY,
                sha256 TEXT NOT NULL UNIQUE,
                relative_path TEXT NOT NULL,
                data TEXT NOT NULL
            );
            """
        )
        conn.execute(
            """
            INSERT INTO artifacts (artifact_id, sha256, relative_path, data)
            VALUES (?, ?, ?, ?)
            """,
            (
                legacy_record.artifact_id,
                legacy_record.sha256,
                legacy_record.relative_path,
                legacy_record.to_json(),
            ),
        )

    store = FilesystemArtifactStore(root)
    try:
        assert store.get_artifact_record_by_id(legacy_record.artifact_id) == legacy_record
        duplicated = store.put_bytes(
            payload,
            content_type="text/plain",
            tool_name="new-tool",
            metadata={"capture": "new"},
        )
        records = store.list_artifact_records_by_sha256(digest)
        assert duplicated.artifact_id != legacy_record.artifact_id
        assert duplicated.sha256 == legacy_record.sha256
        assert duplicated.capture_state.value == "committed"
        assert records == (legacy_record, duplicated)
        assert _read_schema_version(index_path) == 2
    finally:
        store.close()


def test_metadata_store_loads_supported_fixture_revision_v1_and_upgrades_to_current(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "metadata-v1.sqlite"
    _load_sql_fixture(db_path, "metadata/v1.sql")

    store = SqliteMetadataStore(db_path, event_log_path=tmp_path / "events-v1.jsonl")
    try:
        event = store.get_event("evt_fixture")
        assert event is not None
        assert event.event_type == "fixture_event"
        assert _read_schema_version(db_path) == 2
        state = store.get_event_export_state()
        assert state is None
    finally:
        store.close()


def test_metadata_store_loads_supported_fixture_revision_v2_without_mutation(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "metadata-v2.sqlite"
    _load_sql_fixture(db_path, "metadata/v2.sql")

    store = SqliteMetadataStore(db_path, event_log_path=tmp_path / "events-v2.jsonl")
    try:
        assert store.get_event("evt_fixture") is not None
        assert _read_schema_version(db_path) == 2
        state = store.get_event_export_state()
        assert state is not None
        assert state["last_exported_event_id"] == "evt_fixture"
        assert state["full_rebuild_count"] == 1
    finally:
        store.close()


def test_cache_store_loads_supported_fixture_revision_v1(tmp_path: Path) -> None:
    db_path = tmp_path / "cache-v1.sqlite"
    _load_sql_fixture(db_path, "cache/v1.sql")

    store = SqliteCacheStore(db_path)
    try:
        entries = store.list_cache_entries()
        assert len(entries) == 1
        assert entries[0].entry_id == "cache_fixture"
        assert entries[0].value == {"answer": 42}
        assert _read_schema_version(db_path) == 1
    finally:
        store.close()


def test_artifact_store_loads_supported_fixture_revision_v1_and_upgrades_to_current(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifacts-v1"
    db_path = root / "artifact_index.sqlite"
    _load_sql_fixture(db_path, "artifact/v1.sql")
    _write_artifact_blob(
        root,
        "sha256/34f7aed3bc21db8ad882cdc561813afe29bea539f2f951568c38d2c98c2c75ca",
        b"legacy artifact\n",
    )

    store = FilesystemArtifactStore(root)
    try:
        record = store.get_artifact_record_by_id("art_fixture_v1")
        assert record is not None
        assert record.capture_state.value == "committed"
        assert record.metadata["capture"] == "fixture"
        assert _read_schema_version(db_path) == 2
        assert store.read_bytes_by_id("art_fixture_v1") == b"legacy artifact\n"
    finally:
        store.close()


def test_artifact_store_loads_supported_fixture_revision_v2_without_mutation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifacts-v2"
    db_path = root / "artifact_index.sqlite"
    _load_sql_fixture(db_path, "artifact/v2.sql")
    _write_artifact_blob(
        root,
        "sha256/97a5a40a244a405e7f7afc0deba90bf19c65f0defb7595f4ddfb50e122e40bae",
        b"fixture artifact v2\n",
    )

    store = FilesystemArtifactStore(root)
    try:
        record = store.get_artifact_record_by_id("art_fixture_v2")
        assert record is not None
        assert record.capture_group_id == "grp_fixture"
        assert record.actor == "fixture-tests"
        assert record.session_id == "sess_fixture"
        assert _read_schema_version(db_path) == 2
        assert store.read_bytes_by_id("art_fixture_v2") == b"fixture artifact v2\n"
    finally:
        store.close()


def test_metadata_store_rejects_unknown_future_schema_version(tmp_path: Path) -> None:
    db_path = tmp_path / "metadata.sqlite"
    _write_future_schema_version(db_path, 999)

    with pytest.raises(ValueError, match="newer than supported"):
        SqliteMetadataStore(db_path)


def test_cache_store_rejects_unknown_future_schema_version(tmp_path: Path) -> None:
    db_path = tmp_path / "cache.sqlite"
    _write_future_schema_version(db_path, 999)

    with pytest.raises(ValueError, match="newer than supported"):
        SqliteCacheStore(db_path)


def test_artifact_store_rejects_unknown_future_schema_version(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    root.mkdir(parents=True)
    _write_future_schema_version(root / "artifact_index.sqlite", 999)

    with pytest.raises(ValueError, match="newer than supported"):
        FilesystemArtifactStore(root)
