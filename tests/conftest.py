from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from vtm.adapters.embeddings import DeterministicHashEmbeddingAdapter
from vtm.adapters.python_ast import PythonAstSyntaxAdapter
from vtm.adapters.tree_sitter import PythonTreeSitterSyntaxAdapter
from vtm.anchors import CodeAnchor
from vtm.enums import EvidenceKind, MemoryKind, ScopeKind, ValidityStatus
from vtm.evidence import ArtifactRef, EvidenceRef
from vtm.fingerprints import DependencyFingerprint, EnvFingerprint, RepoFingerprint, ToolVersion
from vtm.memory_items import (
    ClaimPayload,
    MemoryItem,
    ProcedurePayload,
    ProcedureStep,
    SummaryCardPayload,
    ValidatorSpec,
    ValidityState,
    VisibilityScope,
)
from vtm.services.embedding_retriever import EmbeddingRetriever
from vtm.services.memory_kernel import TransactionalMemoryKernel
from vtm.services.procedures import CommandProcedureValidator
from vtm.services.retriever import LexicalRetriever
from vtm.services.verifier import BasicVerifier
from vtm.stores.artifact_store import FilesystemArtifactStore
from vtm.stores.cache_store import SqliteCacheStore
from vtm.stores.embedding_store import SqliteEmbeddingIndexStore
from vtm.stores.sqlite_store import SqliteMetadataStore

DEFAULT_PROCEDURE_VALIDATOR = object()


@pytest.fixture
def repo_fp(tmp_path: Path) -> RepoFingerprint:
    return RepoFingerprint(
        repo_root=str(tmp_path),
        branch="main",
        head_commit="abc123",
        tree_digest="tree-1",
        dirty_digest="dirty-1",
    )


@pytest.fixture
def env_fp() -> EnvFingerprint:
    return EnvFingerprint(
        python_version="3.12.8",
        platform="darwin-arm64",
        tool_versions=(ToolVersion(name="pytest", version="8.3.4"),),
    )


@pytest.fixture
def dep_fp(repo_fp: RepoFingerprint, env_fp: EnvFingerprint) -> DependencyFingerprint:
    return DependencyFingerprint(
        repo=repo_fp,
        env=env_fp,
        dependency_ids=("artifact:123",),
        input_digests=("input-1",),
    )


@pytest.fixture
def scope() -> VisibilityScope:
    return VisibilityScope(kind=ScopeKind.BRANCH, scope_id="main")


@pytest.fixture
def artifact_evidence() -> EvidenceRef:
    return EvidenceRef(
        kind=EvidenceKind.ARTIFACT,
        ref_id="artifact:1",
        artifact_ref=ArtifactRef(
            artifact_id="art_existing",
            sha256="deadbeef",
            content_type="text/plain",
        ),
        summary="tool output",
    )


@pytest.fixture
def anchor_evidence() -> EvidenceRef:
    return EvidenceRef(
        kind=EvidenceKind.CODE_ANCHOR,
        ref_id="anchor:1",
        code_anchor=CodeAnchor(
            path="src/example.py",
            symbol="target",
            kind="function",
            language="python",
            ast_digest="ast-1",
            context_digest="ctx-1",
            start_line=10,
            end_line=12,
            start_byte=100,
            end_byte=160,
        ),
        summary="source anchor",
    )


@pytest.fixture
def memory_factory(
    scope: VisibilityScope,
    dep_fp: DependencyFingerprint,
    artifact_evidence: EvidenceRef,
) -> Callable[..., MemoryItem]:
    def _make(
        *,
        title: str = "Parser claim",
        summary: str = "Parser output is stable",
        evidence: tuple[EvidenceRef, ...] | None = None,
        validity_status: ValidityStatus = ValidityStatus.VERIFIED,
        dependency: DependencyFingerprint | None = None,
        tags: tuple[str, ...] = ("parser",),
    ) -> MemoryItem:
        status = validity_status
        dependency_fingerprint = dependency
        if (
            status in {ValidityStatus.VERIFIED, ValidityStatus.RELOCATED}
            and dependency_fingerprint is None
        ):
            dependency_fingerprint = dep_fp
        evidence_refs = evidence if evidence is not None else (artifact_evidence,)
        return MemoryItem(
            kind=MemoryKind.CLAIM,
            title=title,
            summary=summary,
            payload=ClaimPayload(claim=summary),
            evidence=evidence_refs,
            tags=tags,
            visibility=scope,
            validity=ValidityState(status=status, dependency_fingerprint=dependency_fingerprint),
        )

    return _make


@pytest.fixture
def procedure_factory(
    scope: VisibilityScope,
    dep_fp: DependencyFingerprint,
    artifact_evidence: EvidenceRef,
) -> Callable[..., MemoryItem]:
    def _make(
        *,
        title: str = "Parser procedure",
        summary: str = "Run the parser validation flow",
        goal: str = "Run the parser validation flow",
        steps: tuple[ProcedureStep, ...] | None = None,
        validator: ValidatorSpec | None | object = DEFAULT_PROCEDURE_VALIDATOR,
        evidence: tuple[EvidenceRef, ...] | None = None,
        validity_status: ValidityStatus = ValidityStatus.PENDING,
        dependency: DependencyFingerprint | None = None,
        tags: tuple[str, ...] = ("procedure",),
        metadata: dict[str, object] | None = None,
    ) -> MemoryItem:
        status = validity_status
        dependency_fingerprint = dependency
        if (
            status in {ValidityStatus.VERIFIED, ValidityStatus.RELOCATED}
            and dependency_fingerprint is None
        ):
            dependency_fingerprint = dep_fp

        procedure_steps = steps or (ProcedureStep(order=0, instruction="Run parser"),)
        evidence_refs = evidence
        if evidence_refs is None:
            evidence_refs = (
                (artifact_evidence,)
                if status in {ValidityStatus.VERIFIED, ValidityStatus.RELOCATED}
                else ()
            )

        return MemoryItem(
            kind=MemoryKind.PROCEDURE,
            title=title,
            summary=summary,
            payload=ProcedurePayload(
                goal=goal,
                steps=procedure_steps,
                validator=(
                    ValidatorSpec(
                        name="parser-check",
                        kind="command",
                        config={"command": ["python3", "-c", "print('ok')"]},
                    )
                    if validator is DEFAULT_PROCEDURE_VALIDATOR
                    else validator
                ),
            ),
            evidence=evidence_refs,
            tags=tags,
            visibility=scope,
            validity=ValidityState(status=status, dependency_fingerprint=dependency_fingerprint),
            metadata=dict(metadata or {}),
        )

    return _make


@pytest.fixture
def summary_card(scope: VisibilityScope, artifact_evidence: EvidenceRef) -> MemoryItem:
    return MemoryItem(
        kind=MemoryKind.SUMMARY_CARD,
        title="Parser summary card",
        summary="Summarized parser state",
        payload=SummaryCardPayload(
            summary="Summarized parser state",
            supporting_memory_ids=("mem_a",),
        ),
        evidence=(artifact_evidence,),
        visibility=scope,
    )


@pytest.fixture
def metadata_store(tmp_path: Path) -> SqliteMetadataStore:
    store = SqliteMetadataStore(
        tmp_path / "metadata.sqlite",
        event_log_path=tmp_path / "events.jsonl",
    )
    yield store
    store.close()


@pytest.fixture
def artifact_store(tmp_path: Path) -> FilesystemArtifactStore:
    store = FilesystemArtifactStore(tmp_path / "artifacts")
    yield store
    store.close()


@pytest.fixture
def cache_store(tmp_path: Path, metadata_store: SqliteMetadataStore) -> SqliteCacheStore:
    store = SqliteCacheStore(tmp_path / "cache.sqlite", event_store=metadata_store)
    yield store
    store.close()


@pytest.fixture
def embedding_store(tmp_path: Path) -> SqliteEmbeddingIndexStore:
    store = SqliteEmbeddingIndexStore(tmp_path / "embeddings.sqlite")
    yield store
    store.close()


@pytest.fixture
def kernel(
    metadata_store: SqliteMetadataStore,
    artifact_store: FilesystemArtifactStore,
    cache_store: SqliteCacheStore,
) -> TransactionalMemoryKernel:
    anchor_builder = PythonTreeSitterSyntaxAdapter(fallback=PythonAstSyntaxAdapter())
    return TransactionalMemoryKernel(
        metadata_store=metadata_store,
        event_store=metadata_store,
        artifact_store=artifact_store,
        cache_store=cache_store,
        verifier=BasicVerifier(relocator=anchor_builder),
        retriever=LexicalRetriever(metadata_store),
        anchor_adapter=anchor_builder,
        procedure_validator=CommandProcedureValidator(artifact_store),
    )


@pytest.fixture
def embedding_retriever(
    metadata_store: SqliteMetadataStore,
    embedding_store: SqliteEmbeddingIndexStore,
) -> EmbeddingRetriever:
    return EmbeddingRetriever(
        metadata_store,
        embedding_store,
        DeterministicHashEmbeddingAdapter(),
    )
