from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from vtm.adapters.git import GitRepoFingerprintCollector
from vtm.adapters.runtime import RuntimeEnvFingerprintCollector
from vtm.enums import ArtifactCaptureState, EvidenceBudget, EvidenceKind, ValidityStatus
from vtm.memory_items import ValidatorSpec
from vtm.retrieval import RetrieveRequest
from vtm.services.fingerprints import DependencyFingerprintBuilder
from vtm.services.procedures import CommandProcedureValidator


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


def test_command_validator_success_captures_artifacts(
    tmp_path: Path,
    artifact_store,
    procedure_factory,
) -> None:
    validator = CommandProcedureValidator(artifact_store)
    procedure = procedure_factory(
        evidence=(),
        validator=ValidatorSpec(
            name="echo-check",
            kind="command",
            config={"command": ["python3", "-c", "print('ok')"]},
        ),
    )

    result = validator.validate(procedure, repo_root=str(tmp_path))

    assert result.success is True
    assert result.exit_code == 0
    assert result.status is ValidityStatus.VERIFIED
    assert artifact_store.get_artifact_record_by_id(result.stdout_artifact_id) is not None
    assert artifact_store.read_bytes_by_id(result.stdout_artifact_id) == b"ok\n"
    assert artifact_store.get_artifact_record_by_id(result.stderr_artifact_id) is not None


def test_command_validator_commits_captured_artifacts(
    tmp_path: Path,
    artifact_store,
    procedure_factory,
) -> None:
    validator = CommandProcedureValidator(artifact_store)
    procedure = procedure_factory(
        evidence=(),
        validator=ValidatorSpec(
            name="echo-check",
            kind="command",
            config={"command": ["python3", "-c", "print('ok')"]},
        ),
    )

    result = validator.validate(procedure, repo_root=str(tmp_path))
    stdout_record = artifact_store.get_artifact_record_by_id(result.stdout_artifact_id)
    stderr_record = artifact_store.get_artifact_record_by_id(result.stderr_artifact_id)

    assert stdout_record is not None
    assert stderr_record is not None
    assert stdout_record.capture_state is ArtifactCaptureState.COMMITTED
    assert stderr_record.capture_state is ArtifactCaptureState.COMMITTED
    assert artifact_store.audit_integrity().prepared_artifact_ids == ()


def test_command_validator_exit_code_mismatch_refutes(
    tmp_path: Path,
    artifact_store,
    procedure_factory,
) -> None:
    validator = CommandProcedureValidator(artifact_store)
    procedure = procedure_factory(
        evidence=(),
        validator=ValidatorSpec(
            name="failing-check",
            kind="command",
            config={"command": ["python3", "-c", "import sys; sys.exit(7)"]},
        ),
    )

    result = validator.validate(procedure, repo_root=str(tmp_path))

    assert result.success is False
    assert result.exit_code == 7
    assert result.status is ValidityStatus.REFUTED


def test_command_validator_rejects_malformed_config(
    artifact_store,
    procedure_factory,
) -> None:
    validator = CommandProcedureValidator(artifact_store)
    procedure = procedure_factory(
        evidence=(),
        validator=ValidatorSpec(
            name="broken-check",
            kind="command",
            config={"command": "python3 -V"},
        ),
    )

    with pytest.raises(ValueError, match="command list\\[str\\]"):
        validator.validate(procedure)


def test_command_validator_timeout_marks_unknown_and_records_metadata(
    tmp_path: Path,
    artifact_store,
    procedure_factory,
) -> None:
    validator = CommandProcedureValidator(artifact_store)
    procedure = procedure_factory(
        evidence=(),
        validator=ValidatorSpec(
            name="slow-check",
            kind="command",
            config={
                "command": [sys.executable, "-c", "import time; time.sleep(1)"],
                "timeout_seconds": 0.1,
            },
        ),
    )

    result = validator.validate(procedure, repo_root=str(tmp_path))

    assert result.success is False
    assert result.exit_code is None
    assert result.status is ValidityStatus.UNKNOWN
    assert "timed out" in result.reason
    assert result.metadata["timed_out"] is True
    assert artifact_store.read_bytes_by_id(result.stdout_artifact_id) == b""
    assert artifact_store.read_bytes_by_id(result.stderr_artifact_id) == b""


def test_command_validator_truncates_streams_and_records_metadata(
    tmp_path: Path,
    artifact_store,
    procedure_factory,
) -> None:
    validator = CommandProcedureValidator(artifact_store)
    procedure = procedure_factory(
        evidence=(),
        validator=ValidatorSpec(
            name="truncate-check",
            kind="command",
            config={
                "command": [
                    sys.executable,
                    "-c",
                    (
                        "import sys; "
                        "sys.stdout.write('abcdefghij'); "
                        "sys.stderr.write('klmnopqrst')"
                    ),
                ],
                "max_output_bytes": 4,
            },
        ),
    )

    result = validator.validate(procedure, repo_root=str(tmp_path))

    assert result.success is True
    assert result.status is ValidityStatus.VERIFIED
    assert result.metadata["stdout_truncated"] is True
    assert result.metadata["stderr_truncated"] is True
    assert artifact_store.read_bytes_by_id(result.stdout_artifact_id) == b"abcd"
    assert artifact_store.read_bytes_by_id(result.stderr_artifact_id) == b"klmn"


def test_command_validator_applies_env_allowlist_and_denylist(
    tmp_path: Path,
    artifact_store,
    procedure_factory,
    monkeypatch,
) -> None:
    monkeypatch.setenv("KEEP_ME", "present")
    monkeypatch.setenv("DROP_ME", "hidden")
    validator = CommandProcedureValidator(artifact_store)
    procedure = procedure_factory(
        evidence=(),
        validator=ValidatorSpec(
            name="env-check",
            kind="command",
            config={
                "command": [
                    sys.executable,
                    "-c",
                    (
                        "import os; "
                        "print("
                        "f\"{os.environ.get('KEEP_ME')}|"
                        "{os.environ.get('DROP_ME')}|"
                        "{os.environ.get('ADDED')}\""
                        ")"
                    ),
                ],
                "env": {"ADDED": "from-config"},
                "env_allowlist": ["KEEP_ME", "DROP_ME"],
                "env_denylist": ["DROP_ME"],
            },
        ),
    )

    result = validator.validate(procedure, repo_root=str(tmp_path))

    assert result.success is True
    assert result.status is ValidityStatus.VERIFIED
    assert (
        artifact_store.read_bytes_by_id(result.stdout_artifact_id)
        == b"present|None|from-config\n"
    )


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (
            {
                "command": [sys.executable, "-c", "print('ok')"],
                "timeout_seconds": 0,
            },
            "timeout_seconds",
        ),
        (
            {
                "command": [sys.executable, "-c", "print('ok')"],
                "max_output_bytes": 0,
            },
            "max_output_bytes",
        ),
        (
            {
                "command": [sys.executable, "-c", "print('ok')"],
                "env_allowlist": "PATH",
            },
            "env_allowlist",
        ),
        (
            {
                "command": [sys.executable, "-c", "print('ok')"],
                "env_denylist": [123],
            },
            "env_denylist",
        ),
    ],
)
def test_command_validator_rejects_malformed_extended_config(
    artifact_store,
    procedure_factory,
    config,
    message,
) -> None:
    validator = CommandProcedureValidator(artifact_store)
    procedure = procedure_factory(
        evidence=(),
        validator=ValidatorSpec(name="broken-extended-check", kind="command", config=config),
    )

    with pytest.raises(ValueError, match=message):
        validator.validate(procedure)


def test_validate_procedure_promotes_pending_procedure_and_attaches_evidence(
    tmp_path: Path,
    kernel,
    metadata_store,
    artifact_store,
    procedure_factory,
) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    procedure = procedure_factory(
        title="Validated procedure",
        summary="Validated procedure summary",
        goal="Run the validated procedure",
        evidence=(),
        validator=ValidatorSpec(
            name="validated-check",
            kind="command",
            config={"command": ["python3", "-c", "print('validated')"]},
        ),
        validity_status=ValidityStatus.PENDING,
    )
    metadata_store.save_memory_item(procedure)

    result = kernel.validate_procedure(procedure.memory_id, repo_root=str(repo))
    updated = metadata_store.get_memory_item(procedure.memory_id)

    assert updated is not None
    assert result.status is ValidityStatus.VERIFIED
    assert updated.validity.status is ValidityStatus.VERIFIED
    assert updated.validity.dependency_fingerprint is not None
    assert updated.stats.verification_count == 1
    assert updated.metadata["latest_procedure_validation"]["success"] is True
    assert (
        artifact_store.get_artifact_record_by_id(result.stdout_artifact_id).capture_state
        is ArtifactCaptureState.COMMITTED
    )
    assert (
        artifact_store.get_artifact_record_by_id(result.stderr_artifact_id).capture_state
        is ArtifactCaptureState.COMMITTED
    )

    expanded = kernel.expand(procedure.memory_id)
    artifact_ids = {
        evidence.artifact_ref.artifact_id
        for evidence in expanded
        if evidence.kind is EvidenceKind.ARTIFACT and evidence.artifact_ref is not None
    }
    assert artifact_ids == {result.stdout_artifact_id, result.stderr_artifact_id}
    assert {
        event.event_type for event in metadata_store.list_events()
    } >= {"procedure_validated", "memory_expanded"}


def test_validate_procedure_failure_modes_update_memory_status(
    tmp_path: Path,
    kernel,
    metadata_store,
    procedure_factory,
) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)

    refuted = procedure_factory(
        title="Refuted procedure",
        summary="Refuted procedure summary",
        goal="Run the refuted procedure",
        evidence=(),
        validator=ValidatorSpec(
            name="refuted-check",
            kind="command",
            config={"command": ["python3", "-c", "import sys; sys.exit(4)"]},
        ),
    )
    unknown = procedure_factory(
        title="Unknown procedure",
        summary="Unknown procedure summary",
        goal="Run the unknown procedure",
        evidence=(),
        validator=ValidatorSpec(
            name="unknown-check",
            kind="command",
            config={"command": ["definitely-not-a-real-command-for-vtm"]},
        ),
    )
    metadata_store.save_memory_item(refuted)
    metadata_store.save_memory_item(unknown)

    refuted_result = kernel.validate_procedure(refuted.memory_id, repo_root=str(repo))
    unknown_result = kernel.validate_procedure(unknown.memory_id)

    refuted_memory = metadata_store.get_memory_item(refuted.memory_id)
    unknown_memory = metadata_store.get_memory_item(unknown.memory_id)

    assert refuted_memory is not None
    assert unknown_memory is not None
    assert refuted_result.status is ValidityStatus.REFUTED
    assert refuted_memory.validity.status is ValidityStatus.REFUTED
    assert unknown_result.status is ValidityStatus.UNKNOWN
    assert unknown_memory.validity.status is ValidityStatus.UNKNOWN


def test_validate_procedure_downgrades_success_when_dependency_refresh_fails(
    kernel,
    metadata_store,
    procedure_factory,
    dep_fp,
) -> None:
    stale_dependency = dep_fp.model_copy(
        update={
            "repo": dep_fp.repo.model_copy(update={"repo_root": "/definitely/missing/path"}),
        }
    )
    procedure = procedure_factory(
        title="Stale dependency procedure",
        summary="Procedure with stale dependency fingerprint",
        goal="Run the stale dependency procedure",
        evidence=(),
        validator=ValidatorSpec(
            name="validated-check",
            kind="command",
            config={"command": ["python3", "-c", "print('validated')"]},
        ),
        validity_status=ValidityStatus.PENDING,
        dependency=stale_dependency,
    )
    metadata_store.save_memory_item(procedure)

    result = kernel.validate_procedure(procedure.memory_id)
    updated = metadata_store.get_memory_item(procedure.memory_id)

    assert updated is not None
    assert result.exit_code == 0
    assert result.success is False
    assert result.status is ValidityStatus.UNKNOWN
    assert result.metadata["dependency_fingerprint_refresh_failed"] is True
    assert "dependency_fingerprint_error" in result.metadata
    assert updated.validity.status is ValidityStatus.UNKNOWN
    assert updated.validity.dependency_fingerprint == stale_dependency
    assert updated.metadata["latest_procedure_validation"]["success"] is False
    assert (
        updated.metadata["latest_procedure_validation"]["metadata"][
            "dependency_fingerprint_refresh_failed"
        ]
        is True
    )


def test_promote_to_procedure_records_lineage(
    kernel,
    metadata_store,
    memory_factory,
    procedure_factory,
) -> None:
    source_a = memory_factory(title="Source claim A", summary="Source claim summary A")
    source_b = memory_factory(title="Source claim B", summary="Source claim summary B")
    metadata_store.save_memory_item(source_a)
    metadata_store.save_memory_item(source_b)

    promoted = kernel.promote_to_procedure(
        (source_a.memory_id, source_b.memory_id),
        procedure_factory(
            title="Promoted procedure",
            summary="Promoted from verified sources",
            goal="Run the promoted procedure",
            validator=None,
            evidence=(),
        ),
    )

    assert promoted.validity.status is ValidityStatus.VERIFIED
    assert promoted.validity.dependency_fingerprint == source_a.validity.dependency_fingerprint
    promoted_sources = {
        evidence.memory_id for evidence in promoted.evidence if evidence.memory_id is not None
    }
    assert promoted_sources == {source_a.memory_id, source_b.memory_id}
    lineage_edges = metadata_store.list_lineage_edges(child_id=promoted.memory_id)
    assert {edge.parent_id for edge in lineage_edges} == {source_a.memory_id, source_b.memory_id}


def test_procedure_end_to_end_promotion_validation_and_retrieval(
    tmp_path: Path,
    kernel,
    metadata_store,
    scope,
    memory_factory,
    procedure_factory,
) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    dependency = DependencyFingerprintBuilder(
        repo_collector=GitRepoFingerprintCollector(),
        env_collector=RuntimeEnvFingerprintCollector(
            python_version="3.12.9",
            platform_name="test-os-x86_64",
        ),
    ).build(
        str(repo),
        dependency_ids=("memory:source-a", "memory:source-b"),
        input_digests=("promotion",),
    )

    source_a = memory_factory(
        title="Source fact A",
        summary="Verified source fact A",
        dependency=dependency,
    )
    source_b = memory_factory(
        title="Source fact B",
        summary="Verified source fact B",
        dependency=dependency,
    )
    metadata_store.save_memory_item(source_a)
    metadata_store.save_memory_item(source_b)

    promoted = kernel.promote_to_procedure(
        (source_a.memory_id, source_b.memory_id),
        procedure_factory(
            title="Runner procedure",
            summary="Runner procedure summary",
            goal="Execute the runner procedure",
            validator=ValidatorSpec(
                name="runner-check",
                kind="command",
                config={"command": ["python3", "-c", "print('runner ok')"]},
            ),
            evidence=(),
            validity_status=ValidityStatus.PENDING,
        ),
    )
    validation = kernel.validate_procedure(promoted.memory_id, repo_root=str(repo))

    result = kernel.retrieve(
        RetrieveRequest(
            query="runner",
            scopes=(scope,),
            evidence_budget=EvidenceBudget.SUMMARY_FIRST,
        )
    )

    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert candidate.memory.memory_id == promoted.memory_id
    assert candidate.raw_evidence_available is True
    assert candidate.evidence == ()
    assert candidate.explanation.metadata["procedure_has_validator"] is True
    assert candidate.explanation.metadata["procedure_validator_kind"] == "command"
    assert candidate.explanation.metadata["latest_validation_status"] == "verified"
    assert candidate.explanation.metadata["latest_validation_success"] is True

    expanded = kernel.expand(promoted.memory_id)
    memory_refs = {
        evidence.memory_id
        for evidence in expanded
        if evidence.kind is EvidenceKind.MEMORY and evidence.memory_id is not None
    }
    artifact_ids = {
        evidence.artifact_ref.artifact_id
        for evidence in expanded
        if evidence.kind is EvidenceKind.ARTIFACT and evidence.artifact_ref is not None
    }

    assert memory_refs == {source_a.memory_id, source_b.memory_id}
    assert artifact_ids == {validation.stdout_artifact_id, validation.stderr_artifact_id}
    assert {
        event.event_type for event in metadata_store.list_events()
    } >= {
        "procedure_promoted",
        "procedure_validated",
        "memory_retrieved",
        "memory_expanded",
    }
