"""Filesystem blob storage backed by a SQLite artifact index."""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from vtm.artifacts import ArtifactIntegrityReport, ArtifactRecord
from vtm.base import utc_now
from vtm.enums import ArtifactCaptureState
from vtm.ids import new_artifact_id
from vtm.stores._sqlite_schema import (
    ensure_schema_tracking_tables,
    read_schema_version,
    record_schema_migration,
)
from vtm.stores.migrations.artifact import (
    ARTIFACT_SCHEMA_VERSION,
    apply_artifact_migrations,
)


class FilesystemArtifactStore:
    """Artifact store that persists blobs on disk and metadata in SQLite."""

    def __init__(self, root: str | Path) -> None:
        """Open or initialize the artifact root and index."""
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._blob_root = self._root / "sha256"
        self._blob_root.mkdir(parents=True, exist_ok=True)
        self._index_path = self._root / "artifact_index.sqlite"
        self._conn = sqlite3.connect(self._index_path)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def close(self) -> None:
        """Close the artifact index connection."""
        self._conn.close()

    def _init_schema(self) -> None:
        ensure_schema_tracking_tables(self._conn)
        current_version = read_schema_version(self._conn)
        if current_version > ARTIFACT_SCHEMA_VERSION:
            raise ValueError(
                f"artifact store schema version {current_version} is newer than supported "
                f"{ARTIFACT_SCHEMA_VERSION}"
            )
        with self._conn:
            apply_artifact_migrations(self._conn, current_version)
            for version in range(current_version + 1, ARTIFACT_SCHEMA_VERSION + 1):
                record_schema_migration(self._conn, version)

    def prepare_bytes(
        self,
        data: bytes,
        *,
        content_type: str = "application/octet-stream",
        tool_name: str | None = None,
        tool_version: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        artifact_id: str | None = None,
        capture_group_id: str | None = None,
        actor: str = "system",
        session_id: str | None = None,
    ) -> ArtifactRecord:
        digest = hashlib.sha256(data).hexdigest()
        if artifact_id is None:
            artifact_id = new_artifact_id()
        elif self.get_artifact_record_by_id(artifact_id) is not None:
            raise ValueError(f"artifact id already exists: {artifact_id}")

        blob_path = self._blob_root / digest
        if blob_path.exists():
            current = blob_path.read_bytes()
            if current != data:
                raise ValueError(f"content mismatch for artifact digest {digest}")
        else:
            blob_path.write_bytes(data)

        record = ArtifactRecord(
            artifact_id=artifact_id,
            sha256=digest,
            relative_path=f"sha256/{digest}",
            size_bytes=len(data),
            content_type=content_type,
            tool_name=tool_name,
            tool_version=tool_version,
            capture_state=ArtifactCaptureState.PREPARED,
            capture_group_id=capture_group_id,
            actor=actor,
            session_id=session_id,
            metadata=dict(metadata or {}),
        )
        self._save_record(record)
        return record

    def commit_artifact(self, artifact_id: str) -> ArtifactRecord:
        record = self._require_artifact_record(artifact_id)
        if record.capture_state is ArtifactCaptureState.COMMITTED:
            return record
        committed = record.model_copy(
            update={
                "capture_state": ArtifactCaptureState.COMMITTED,
                "committed_at": utc_now(),
                "abandoned_at": None,
            }
        )
        self._save_record(committed)
        return committed

    def abandon_artifact(
        self,
        artifact_id: str,
        *,
        reason: str | None = None,
    ) -> ArtifactRecord:
        record = self._require_artifact_record(artifact_id)
        metadata = dict(record.metadata)
        if reason is not None:
            metadata["abandon_reason"] = reason
        abandoned = record.model_copy(
            update={
                "capture_state": ArtifactCaptureState.ABANDONED,
                "abandoned_at": utc_now(),
                "metadata": metadata,
            }
        )
        self._save_record(abandoned)
        return abandoned

    def put_bytes(
        self,
        data: bytes,
        *,
        content_type: str = "application/octet-stream",
        tool_name: str | None = None,
        tool_version: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        artifact_id: str | None = None,
    ) -> ArtifactRecord:
        prepared = self.prepare_bytes(
            data,
            content_type=content_type,
            tool_name=tool_name,
            tool_version=tool_version,
            metadata=metadata,
            artifact_id=artifact_id,
            capture_group_id=artifact_id,
        )
        return self.commit_artifact(prepared.artifact_id)

    def get_artifact_record_by_id(self, artifact_id: str) -> ArtifactRecord | None:
        row = self._conn.execute(
            "SELECT data FROM artifacts WHERE artifact_id = ?",
            (artifact_id,),
        ).fetchone()
        if row is None:
            return None
        return ArtifactRecord.from_json(row["data"])

    def get_artifact_record_by_sha256(self, sha256: str) -> ArtifactRecord | None:
        row = self._conn.execute(
            """
            SELECT data
            FROM artifacts
            WHERE sha256 = ?
            ORDER BY
                CASE capture_state
                    WHEN 'committed' THEN 0
                    WHEN 'prepared' THEN 1
                    ELSE 2
                END ASC,
                rowid DESC
            LIMIT 1
            """,
            (sha256,),
        ).fetchone()
        if row is None:
            return None
        return ArtifactRecord.from_json(row["data"])

    def list_artifact_records_by_sha256(self, sha256: str) -> tuple[ArtifactRecord, ...]:
        rows = self._conn.execute(
            "SELECT data FROM artifacts WHERE sha256 = ? ORDER BY rowid ASC",
            (sha256,),
        ).fetchall()
        return tuple(ArtifactRecord.from_json(row["data"]) for row in rows)

    def list_artifact_records(
        self,
        *,
        capture_state: ArtifactCaptureState | None = None,
    ) -> tuple[ArtifactRecord, ...]:
        sql = "SELECT data FROM artifacts"
        params: list[str] = []
        if capture_state is not None:
            sql += " WHERE capture_state = ?"
            params.append(capture_state.value)
        sql += " ORDER BY rowid ASC"
        rows = self._conn.execute(sql, params).fetchall()
        return tuple(ArtifactRecord.from_json(row["data"]) for row in rows)

    def audit_integrity(self) -> ArtifactIntegrityReport:
        prepared_artifact_ids = tuple(
            record.artifact_id
            for record in self.list_artifact_records(
                capture_state=ArtifactCaptureState.PREPARED,
            )
        )
        committed_missing_blob_artifact_ids: list[str] = []
        referenced_paths: set[str] = set()
        for record in self.list_artifact_records():
            if record.capture_state is ArtifactCaptureState.COMMITTED:
                referenced_paths.add(record.relative_path)
                blob_path = self._root / record.relative_path
                if not blob_path.exists():
                    committed_missing_blob_artifact_ids.append(record.artifact_id)
                continue
            if record.capture_state is ArtifactCaptureState.PREPARED:
                referenced_paths.add(record.relative_path)

        orphaned_blob_paths = tuple(
            sorted(
                str(blob_path.relative_to(self._root))
                for blob_path in self._blob_root.glob("*")
                if str(blob_path.relative_to(self._root)) not in referenced_paths
            )
        )
        return ArtifactIntegrityReport(
            prepared_artifact_ids=prepared_artifact_ids,
            committed_missing_blob_artifact_ids=tuple(committed_missing_blob_artifact_ids),
            orphaned_blob_paths=orphaned_blob_paths,
        )

    def abandon_stale_prepared_artifacts(self) -> Sequence[ArtifactRecord]:
        abandoned: list[ArtifactRecord] = []
        for record in self.list_artifact_records(capture_state=ArtifactCaptureState.PREPARED):
            abandoned.append(
                self.abandon_artifact(
                    record.artifact_id,
                    reason="janitor_abandoned_prepared_capture",
                )
            )
        return tuple(abandoned)

    def cleanup_orphaned_blobs(self) -> Sequence[str]:
        referenced = {
            record.relative_path
            for record in self.list_artifact_records()
            if record.capture_state is not ArtifactCaptureState.ABANDONED
        }
        removed: list[str] = []
        for blob_path in sorted(self._blob_root.glob("*")):
            relative_path = str(blob_path.relative_to(self._root))
            if relative_path in referenced:
                continue
            blob_path.unlink(missing_ok=True)
            removed.append(relative_path)
        return tuple(removed)

    def read_bytes_by_id(self, artifact_id: str) -> bytes | None:
        record = self.get_artifact_record_by_id(artifact_id)
        if record is None:
            return None
        blob_path = self._root / record.relative_path
        if not blob_path.exists():
            return None
        return blob_path.read_bytes()

    def __enter__(self) -> FilesystemArtifactStore:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _save_record(self, record: ArtifactRecord) -> None:
        self._conn.execute(
            """
            INSERT INTO artifacts (
                artifact_id, sha256, relative_path, capture_state, capture_group_id,
                committed_at, abandoned_at, actor, session_id, data
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(artifact_id) DO UPDATE SET
                sha256 = excluded.sha256,
                relative_path = excluded.relative_path,
                capture_state = excluded.capture_state,
                capture_group_id = excluded.capture_group_id,
                committed_at = excluded.committed_at,
                abandoned_at = excluded.abandoned_at,
                actor = excluded.actor,
                session_id = excluded.session_id,
                data = excluded.data
            """,
            (
                record.artifact_id,
                record.sha256,
                record.relative_path,
                record.capture_state.value,
                record.capture_group_id,
                record.committed_at.isoformat() if record.committed_at else None,
                record.abandoned_at.isoformat() if record.abandoned_at else None,
                record.actor,
                record.session_id,
                record.to_json(),
            ),
        )
        self._conn.commit()

    def _require_artifact_record(self, artifact_id: str) -> ArtifactRecord:
        record = self.get_artifact_record_by_id(artifact_id)
        if record is None:
            raise KeyError(f"unknown artifact record: {artifact_id}")
        return record
