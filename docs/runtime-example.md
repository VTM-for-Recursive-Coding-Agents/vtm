# Runtime Example

```python
import subprocess
import tempfile
from pathlib import Path

from vtm import ClaimPayload, MemoryItem, RetrieveRequest, ValidityState, ValidityStatus, VisibilityScope
from vtm.adapters import (
    GitRepoFingerprintCollector,
    PythonAstSyntaxAdapter,
    PythonTreeSitterSyntaxAdapter,
    RuntimeEnvFingerprintCollector,
)
from vtm.enums import EvidenceBudget, ScopeKind
from vtm.services import BasicVerifier, DependencyFingerprintBuilder, LexicalRetriever, TransactionalMemoryKernel
from vtm.stores import FilesystemArtifactStore, SqliteCacheStore, SqliteMetadataStore

with tempfile.TemporaryDirectory() as temp_dir:
    repo_root = Path(temp_dir) / "repo"
    repo_root.mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=repo_root, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "VTM Example"], cwd=repo_root, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "vtm@example.com"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    (repo_root / "module.py").write_text(
        "def helper():\n"
        "    return 1\n\n"
        "def target():\n"
        "    return helper()\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "module.py"], cwd=repo_root, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=repo_root, check=True, capture_output=True)

    metadata = SqliteMetadataStore(
        repo_root / ".vtm" / "metadata.sqlite",
        event_log_path=repo_root / ".vtm" / "events.jsonl",
    )
    artifacts = FilesystemArtifactStore(repo_root / ".vtm" / "artifacts")
    cache = SqliteCacheStore(repo_root / ".vtm" / "cache.sqlite", event_store=metadata)
    anchor_adapter = PythonTreeSitterSyntaxAdapter(fallback=PythonAstSyntaxAdapter())
    kernel = TransactionalMemoryKernel(
        metadata_store=metadata,
        event_store=metadata,
        artifact_store=artifacts,
        cache_store=cache,
        verifier=BasicVerifier(relocator=anchor_adapter),
        retriever=LexicalRetriever(metadata),
        anchor_adapter=anchor_adapter,
    )

    fingerprints = DependencyFingerprintBuilder(
        repo_collector=GitRepoFingerprintCollector(),
        env_collector=RuntimeEnvFingerprintCollector(),
    ).build(
        str(repo_root),
        dependency_ids=("tool:lint",),
        input_digests=("stdin:empty",),
    )

    artifact = kernel.capture_artifact(
        b"flake8 output\n",
        content_type="text/plain",
        tool_name="flake8",
        tool_version="7.1.0",
        metadata={"command": "flake8 ."},
    )
    integrity = artifacts.audit_integrity()
    evidence = kernel.artifact_evidence(
        artifact,
        label="flake8-output",
        summary="Captured linter output",
    )
    anchor = kernel.build_code_anchor(str(repo_root / "module.py"), "target")
    anchor_evidence = kernel.anchor_evidence(
        anchor,
        label="target-anchor",
        summary="Captured code anchor",
    )

    scope = VisibilityScope(kind=ScopeKind.BRANCH, scope_id="main")
    tx = kernel.begin_transaction(scope)
    memory = MemoryItem(
        kind="claim",
        title="Lint output for main",
        summary="Latest flake8 run for the branch",
        payload=ClaimPayload(claim="Latest flake8 run for the branch"),
        evidence=(evidence, anchor_evidence),
        visibility=scope,
        validity=ValidityState(
            status=ValidityStatus.VERIFIED,
            dependency_fingerprint=fingerprints,
        ),
    )
    staged = kernel.stage_memory_item(tx.tx_id, memory)
    kernel.commit_transaction(tx.tx_id)

    cards = kernel.retrieve(
        RetrieveRequest(
            query="flake8",
            scopes=(scope,),
            evidence_budget=EvidenceBudget.SUMMARY_FIRST,
        )
    )
    raw_evidence = kernel.expand(staged.memory_id)

    metadata.export_events_to_jsonl()
    metadata.rebuild_events_jsonl()

    (repo_root / "module.py").write_text(
        "def helper():\n"
        "    return 1\n\n\n"
        "def target():\n"
        "    return helper()\n",
        encoding="utf-8",
    )
    updated_fingerprints = DependencyFingerprintBuilder(
        repo_collector=GitRepoFingerprintCollector(),
        env_collector=RuntimeEnvFingerprintCollector(),
    ).build(
        str(repo_root),
        dependency_ids=("tool:lint",),
        input_digests=("stdin:empty",),
    )
    updated_memory, verification = kernel.verify_memory(staged.memory_id, updated_fingerprints)

    assert len(cards.candidates) == 1
    assert len(raw_evidence) == 2
    assert integrity.prepared_artifact_ids == ()
    assert integrity.committed_missing_blob_artifact_ids == ()
    assert integrity.orphaned_blob_paths == ()
    assert updated_memory.validity.status is ValidityStatus.RELOCATED
    assert verification.current_status is ValidityStatus.RELOCATED

    cache.close()
    artifacts.close()
    metadata.close()
```

With code-anchor evidence, a non-semantic source move can now transition the memory to `relocated` with an updated anchor span instead of dropping to `unknown`. Python Tree-sitter is the primary adapter path; Python AST remains as a fallback/parity implementation. The artifact store can also expose a non-mutating integrity report before any janitor cleanup runs.
