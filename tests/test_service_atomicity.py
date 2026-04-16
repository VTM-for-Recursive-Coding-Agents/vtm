from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from vtm.enums import EvidenceBudget, ValidityStatus
from vtm.memory_items import ValidatorSpec
from vtm.retrieval import RetrieveRequest


def _run(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        list(args),
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _init_repo(repo: Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    _run(repo, "git", "init")
    _run(repo, "git", "config", "user.name", "VTM Tests")
    _run(repo, "git", "config", "user.email", "vtm@example.com")
    (repo / "module.py").write_text(
        "def target():\n"
        "    return 'ok'\n",
        encoding="utf-8",
    )
    _run(repo, "git", "add", "module.py")
    _run(repo, "git", "commit", "-m", "initial")


def _fail_event_type(monkeypatch, metadata_store, event_type: str) -> None:
    original_save_event = metadata_store.save_event

    def save_event(event) -> None:
        if event.event_type == event_type:
            raise RuntimeError(f"simulated {event_type} event failure")
        original_save_event(event)

    monkeypatch.setattr(metadata_store, "save_event", save_event)


def test_verify_memory_rolls_back_when_event_persistence_fails(
    kernel,
    metadata_store,
    memory_factory,
    dep_fp,
    monkeypatch,
) -> None:
    item = memory_factory(title="Verify me")
    metadata_store.save_memory_item(item)
    _fail_event_type(monkeypatch, metadata_store, "memory_verified")

    with pytest.raises(RuntimeError, match="simulated memory_verified event failure"):
        kernel.verify_memory(item.memory_id, dep_fp)

    persisted = metadata_store.get_memory_item(item.memory_id)
    assert persisted == item
    assert not any(event.event_type == "memory_verified" for event in metadata_store.list_events())


def test_validate_procedure_rolls_back_metadata_when_event_persistence_fails(
    tmp_path: Path,
    kernel,
    metadata_store,
    artifact_store,
    procedure_factory,
    monkeypatch,
) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    procedure = procedure_factory(
        title="Atomic validation",
        evidence=(),
        validator=ValidatorSpec(
            name="validated-check",
            kind="command",
            config={"command": ["python3", "-c", "print('validated')"]},
        ),
        validity_status=ValidityStatus.PENDING,
    )
    metadata_store.save_memory_item(procedure)
    _fail_event_type(monkeypatch, metadata_store, "procedure_validated")

    with pytest.raises(RuntimeError, match="simulated procedure_validated event failure"):
        kernel.validate_procedure(procedure.memory_id, repo_root=str(repo))

    persisted = metadata_store.get_memory_item(procedure.memory_id)
    assert persisted == procedure
    assert "latest_procedure_validation" not in persisted.metadata
    assert not any(
        event.event_type == "procedure_validated" for event in metadata_store.list_events()
    )
    validation_artifacts = [
        record
        for record in artifact_store.list_artifact_records()
        if record.metadata.get("memory_id") == procedure.memory_id
    ]
    assert len(validation_artifacts) == 2
    assert {record.capture_state.value for record in validation_artifacts} == {"abandoned"}
    assert {
        record.metadata.get("abandon_reason") for record in validation_artifacts
    } == {"procedure_validation_writeback_failed"}
    assert {
        tuple(sorted(record.metadata.get("abandon_provenance", {}).items()))
        for record in validation_artifacts
    } == {
        (("origin", "procedure_validation"), ("stage", "metadata_or_event_writeback")),
    }


def test_promote_to_procedure_rolls_back_when_event_persistence_fails(
    kernel,
    metadata_store,
    memory_factory,
    procedure_factory,
    monkeypatch,
) -> None:
    source_a = memory_factory(title="Atomic source A")
    source_b = memory_factory(title="Atomic source B")
    metadata_store.save_memory_item(source_a)
    metadata_store.save_memory_item(source_b)
    procedure = procedure_factory(
        title="Atomic promoted procedure",
        validator=None,
        evidence=(),
    )
    _fail_event_type(monkeypatch, metadata_store, "procedure_promoted")

    with pytest.raises(RuntimeError, match="simulated procedure_promoted event failure"):
        kernel.promote_to_procedure((source_a.memory_id, source_b.memory_id), procedure)

    assert metadata_store.get_memory_item(procedure.memory_id) is None
    assert metadata_store.list_lineage_edges(child_id=procedure.memory_id) == ()
    assert not any(
        event.event_type == "procedure_promoted" for event in metadata_store.list_events()
    )


def test_retrieve_rolls_back_stats_when_event_persistence_fails(
    kernel,
    metadata_store,
    memory_factory,
    scope,
    monkeypatch,
) -> None:
    first = memory_factory(title="Parser result one", summary="parser result one")
    second = memory_factory(title="Parser result two", summary="parser result two")
    metadata_store.save_memory_item(first)
    metadata_store.save_memory_item(second)
    _fail_event_type(monkeypatch, metadata_store, "memory_retrieved")

    with pytest.raises(RuntimeError, match="simulated memory_retrieved event failure"):
        kernel.retrieve(
            RetrieveRequest(
                query="parser",
                scopes=(scope,),
                evidence_budget=EvidenceBudget.SUMMARY_FIRST,
            )
        )

    persisted_first = metadata_store.get_memory_item(first.memory_id)
    persisted_second = metadata_store.get_memory_item(second.memory_id)
    assert persisted_first is not None
    assert persisted_second is not None
    assert persisted_first.stats.retrieval_count == 0
    assert persisted_second.stats.retrieval_count == 0
    assert not any(
        event.event_type == "memory_retrieved" for event in metadata_store.list_events()
    )


def test_capture_artifact_abandons_record_when_event_writeback_fails(
    kernel,
    metadata_store,
    artifact_store,
    monkeypatch,
) -> None:
    _fail_event_type(monkeypatch, metadata_store, "artifact_captured")

    with pytest.raises(RuntimeError, match="simulated artifact_captured event failure"):
        kernel.capture_artifact(
            b"artifact payload",
            content_type="text/plain",
            tool_name="pytest",
        )

    records = artifact_store.list_artifact_records()
    assert len(records) == 1
    assert records[0].capture_state.value == "abandoned"
    assert records[0].metadata.get("abandon_reason") == "artifact_capture_writeback_failed"
    assert records[0].metadata.get("abandon_provenance") == {
        "origin": "kernel_artifact_capture",
        "stage": "event_writeback",
    }
