from __future__ import annotations

import subprocess
from pathlib import Path

from vtm.adapters.git import GitRepoFingerprintCollector
from vtm.adapters.runtime import RuntimeEnvFingerprintCollector
from vtm.enums import EvidenceBudget, ValidityStatus
from vtm.memory_items import ClaimPayload, MemoryItem, ValidityState
from vtm.retrieval import RetrieveRequest
from vtm.services.fingerprints import DependencyFingerprintBuilder


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
    (repo / "tool.log").write_text("initial\n", encoding="utf-8")
    (repo / "module.py").write_text(
        "def helper():\n"
        "    return 1\n\n"
        "def target():\n"
        "    return helper()\n",
        encoding="utf-8",
    )
    _run(repo, "git", "add", "tool.log", "module.py")
    _run(repo, "git", "commit", "-m", "initial")


def test_kernel_capture_artifact_and_runtime_verification_flow(
    tmp_path: Path,
    kernel,
    metadata_store,
    scope,
) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    fp_builder = DependencyFingerprintBuilder(
        repo_collector=GitRepoFingerprintCollector(),
        env_collector=RuntimeEnvFingerprintCollector(
            python_version="3.12.9",
            platform_name="test-os-x86_64",
        ),
    )
    dependency = fp_builder.build(
        str(repo),
        dependency_ids=("tool:example",),
        input_digests=("input-1",),
    )

    artifact = kernel.capture_artifact(
        b"tool output\n",
        content_type="text/plain",
        tool_name="example-tool",
        tool_version="1.0.0",
        metadata={"command": "example"},
    )
    artifact_evidence = kernel.artifact_evidence(
        artifact,
        label="example-output",
        summary="Captured tool output",
    )
    anchor = kernel.build_code_anchor(str(repo / "module.py"), "target")
    anchor_evidence = kernel.anchor_evidence(
        anchor,
        label="target-anchor",
        summary="Captured code anchor",
    )

    tx = kernel.begin_transaction(scope)
    memory = MemoryItem(
        kind="claim",
        title="Captured parser output",
        summary="Artifact-backed parser output",
        payload=ClaimPayload(claim="Artifact-backed parser output"),
        evidence=(artifact_evidence, anchor_evidence),
        visibility=scope,
        validity=ValidityState(
            status=ValidityStatus.VERIFIED,
            dependency_fingerprint=dependency,
        ),
        tags=("parser", "runtime"),
    )
    staged = kernel.stage_memory_item(tx.tx_id, memory)
    kernel.commit_transaction(tx.tx_id)

    result = kernel.retrieve(
        RetrieveRequest(
            query="parser output",
            scopes=(scope,),
            evidence_budget=EvidenceBudget.SUMMARY_FIRST,
        )
    )
    assert len(result.candidates) == 1
    retrieved = result.candidates[0].memory
    assert retrieved.memory_id == staged.memory_id
    assert retrieved.stats.retrieval_count == 1
    assert result.candidates[0].raw_evidence_available is True
    assert result.candidates[0].evidence == ()

    expanded = kernel.expand(staged.memory_id)
    assert expanded == (artifact_evidence, anchor_evidence)

    module_path = repo / "module.py"
    module_path.write_text(
        "def helper():\n"
        "    return 1\n\n"
        "\n"
        "def target():\n"
        "    return helper()\n",
        encoding="utf-8",
    )
    changed_dependency = fp_builder.build(
        str(repo),
        dependency_ids=("tool:example",),
        input_digests=("input-1",),
    )
    updated_memory, verification = kernel.verify_memory(staged.memory_id, changed_dependency)
    assert verification.current_status is ValidityStatus.RELOCATED
    assert updated_memory.validity.status is ValidityStatus.RELOCATED
    updated_anchor = next(
        evidence.code_anchor
        for evidence in updated_memory.evidence
        if evidence.code_anchor is not None
    )
    assert updated_anchor.start_line == anchor.start_line + 1

    event_types = [event.event_type for event in metadata_store.list_events()]
    assert "artifact_capture_prepared" in event_types
    assert "artifact_captured" in event_types
    assert "anchor_built" in event_types
    assert "memory_retrieved" in event_types
    assert "memory_expanded" in event_types
