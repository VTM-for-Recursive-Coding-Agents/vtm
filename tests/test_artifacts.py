from __future__ import annotations

import pytest


def test_artifact_store_reuses_blob_storage_but_keeps_distinct_capture_records(
    artifact_store,
) -> None:
    first = artifact_store.put_bytes(
        b"parser output",
        content_type="text/plain",
        tool_name="pytest",
        tool_version="8.3.4",
        metadata={"run_id": "one"},
    )
    second = artifact_store.put_bytes(
        b"parser output",
        content_type="text/plain",
        tool_name="pytest",
        tool_version="8.3.4",
        metadata={"run_id": "two"},
    )

    assert first.artifact_id != second.artifact_id
    assert first.sha256 == second.sha256
    assert first.relative_path == second.relative_path
    assert first.metadata["run_id"] == "one"
    assert second.metadata["run_id"] == "two"
    assert artifact_store.get_artifact_record_by_sha256(first.sha256) == second
    assert artifact_store.list_artifact_records_by_sha256(first.sha256) == (first, second)
    assert artifact_store.read_bytes_by_id(first.artifact_id) == b"parser output"
    assert artifact_store.read_bytes_by_id(second.artifact_id) == b"parser output"
    assert first.capture_state.value == "committed"
    assert second.capture_state.value == "committed"


def test_artifact_store_rejects_silent_overwrite(artifact_store) -> None:
    first = artifact_store.put_bytes(b"one", artifact_id="art_fixed")

    with pytest.raises(ValueError):
        artifact_store.put_bytes(b"two", artifact_id=first.artifact_id)


def test_artifact_store_prepared_commit_abandon_and_cleanup(artifact_store) -> None:
    prepared = artifact_store.prepare_bytes(
        b"prepared",
        content_type="text/plain",
        tool_name="pytest",
        capture_group_id="grp_1",
        actor="tests",
        session_id="sess_1",
    )
    assert prepared.capture_state.value == "prepared"
    assert prepared.capture_group_id == "grp_1"
    assert prepared.actor == "tests"
    assert prepared.session_id == "sess_1"

    committed = artifact_store.commit_artifact(prepared.artifact_id)
    assert committed.capture_state.value == "committed"
    assert committed.committed_at is not None

    abandoned = artifact_store.prepare_bytes(
        b"abandon-me",
        content_type="text/plain",
        capture_group_id="grp_2",
    )
    artifact_store.abandon_artifact(
        abandoned.artifact_id,
        reason="test-abandon",
        provenance={"origin": "tests", "stage": "unit"},
    )
    janitor = artifact_store.abandon_stale_prepared_artifacts()
    assert janitor == ()

    audit_before_cleanup = artifact_store.audit_integrity()
    assert audit_before_cleanup.prepared_artifact_ids == ()
    assert audit_before_cleanup.committed_missing_blob_artifact_ids == ()
    assert audit_before_cleanup.orphaned_blob_paths == (abandoned.relative_path,)
    assert audit_before_cleanup.abandoned_artifact_ids_by_reason == {
        "test-abandon": (abandoned.artifact_id,)
    }
    assert audit_before_cleanup.abandoned_artifact_ids_by_origin == {
        "tests": (abandoned.artifact_id,)
    }

    removed = artifact_store.cleanup_orphaned_blobs()
    assert removed == (abandoned.relative_path,)
    assert artifact_store.audit_integrity().orphaned_blob_paths == ()


def test_artifact_store_janitor_abandons_prepared_records(artifact_store) -> None:
    prepared = artifact_store.prepare_bytes(b"pending", capture_group_id="grp_pending")
    abandoned = artifact_store.abandon_stale_prepared_artifacts()
    assert [record.artifact_id for record in abandoned] == [prepared.artifact_id]
    refreshed = artifact_store.get_artifact_record_by_id(prepared.artifact_id)
    assert refreshed is not None
    assert refreshed.capture_state.value == "abandoned"
    assert refreshed.metadata["abandon_reason"] == "janitor_abandoned_prepared_capture"
    assert refreshed.metadata["abandon_provenance"] == {
        "origin": "artifact_janitor",
        "stage": "prepared_capture_cleanup",
    }


def test_artifact_store_audit_reports_prepared_missing_and_orphaned_blobs(
    artifact_store,
) -> None:
    prepared = artifact_store.prepare_bytes(b"prepared-payload", capture_group_id="grp_pending")
    committed = artifact_store.put_bytes(b"committed-payload", content_type="text/plain")

    committed_blob_path = artifact_store._root / committed.relative_path
    committed_blob_path.unlink()
    orphaned_blob_path = artifact_store._blob_root / "deadbeef"
    orphaned_blob_path.write_bytes(b"orphaned")

    audit = artifact_store.audit_integrity()

    assert audit.prepared_artifact_ids == (prepared.artifact_id,)
    assert audit.committed_missing_blob_artifact_ids == (committed.artifact_id,)
    assert audit.orphaned_blob_paths == ("sha256/deadbeef",)


def test_artifact_store_repair_integrity_applies_safe_repairs_and_reports_residuals(
    artifact_store,
) -> None:
    prepared = artifact_store.prepare_bytes(b"prepared-payload", capture_group_id="grp_pending")
    committed = artifact_store.put_bytes(b"committed-payload", content_type="text/plain")
    committed_blob_path = artifact_store._root / committed.relative_path
    committed_blob_path.unlink()
    orphaned_blob_path = artifact_store._blob_root / "deadbeef"
    orphaned_blob_path.write_bytes(b"orphaned")

    repair = artifact_store.repair_integrity()
    prepared_after = artifact_store.get_artifact_record_by_id(prepared.artifact_id)

    assert repair.audit_before.prepared_artifact_ids == (prepared.artifact_id,)
    assert repair.audit_before.committed_missing_blob_artifact_ids == (committed.artifact_id,)
    assert repair.audit_before.orphaned_blob_paths == ("sha256/deadbeef",)
    assert repair.abandoned_prepared_artifact_ids == (prepared.artifact_id,)
    assert repair.removed_orphaned_blob_paths == tuple(
        sorted((prepared.relative_path, "sha256/deadbeef"))
    )
    assert repair.unresolved_committed_missing_blob_artifact_ids == (committed.artifact_id,)
    assert repair.audit_after.prepared_artifact_ids == ()
    assert repair.audit_after.orphaned_blob_paths == ()
    assert repair.audit_after.committed_missing_blob_artifact_ids == (committed.artifact_id,)
    assert prepared_after is not None
    assert prepared_after.capture_state.value == "abandoned"
