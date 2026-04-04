from __future__ import annotations

import importlib

import pytest
from pydantic import ValidationError

from vtm.adapters.rlm import RLMRankedCandidate, RLMRankRequest, RLMRankResponse
from vtm.artifacts import ArtifactRecord
from vtm.benchmarks import (
    BenchmarkManifest,
    BenchmarkRunConfig,
    CodingTaskCase,
    CommitPair,
    RepoSpec,
)
from vtm.cache import CacheEntry, CacheKey
from vtm.consolidation import ConsolidationAction, ConsolidationRunResult
from vtm.embeddings import EmbeddingIndexEntry
from vtm.enums import (
    DetailLevel,
    EvidenceBudget,
    EvidenceKind,
    MemoryKind,
    ValidityStatus,
)
from vtm.events import MemoryEvent
from vtm.evidence import EvidenceRef
from vtm.memory_items import (
    ClaimPayload,
    MemoryItem,
    ProcedurePayload,
    ProcedureStep,
    SummaryCardPayload,
    ValidatorSpec,
    ValidityState,
)
from vtm.retrieval import RetrieveRequest
from vtm.transactions import TransactionRecord
from vtm.verification import ProcedureValidationResult, VerificationResult


def test_package_import_smoke() -> None:
    module = importlib.import_module("vtm")
    assert module.SCHEMA_VERSION == "1.0"


def test_core_models_round_trip(
    repo_fp,
    env_fp,
    dep_fp,
    scope,
    artifact_evidence,
    memory_factory,
) -> None:
    memory = memory_factory()
    validator = ValidatorSpec(
        name="parser-check",
        kind="command",
        config={"command": ["python3", "-c", "print('ok')"]},
    )
    models = [
        repo_fp,
        env_fp,
        dep_fp,
        scope,
        artifact_evidence,
        memory,
        validator,
        ProcedurePayload(
            goal="Run parser check",
            steps=(ProcedureStep(order=0, instruction="Run parser check"),),
            validator=validator,
        ),
        ArtifactRecord(
            sha256="deadbeef",
            relative_path="sha256/deadbeef",
            size_bytes=4,
            content_type="text/plain",
        ),
        TransactionRecord(visibility=scope),
        RetrieveRequest(
            query="parser stable",
            scopes=(scope,),
            detail_level=DetailLevel.SUMMARY,
            evidence_budget=EvidenceBudget.SUMMARY_FIRST,
        ),
        CacheKey.from_parts("tool", {"b": 2, "a": 1}, repo_fp, env_fp),
        MemoryEvent(event_type="memory_staged", memory_id=memory.memory_id),
        VerificationResult(
            memory_id=memory.memory_id,
            previous_status=ValidityStatus.VERIFIED,
            current_status=ValidityStatus.VERIFIED,
            dependency_changed=False,
            updated_validity=ValidityState(
                status=ValidityStatus.VERIFIED,
                dependency_fingerprint=dep_fp,
            ),
            skipped=True,
        ),
        ProcedureValidationResult(
            memory_id=memory.memory_id,
            validator_spec=validator,
            success=True,
            exit_code=0,
            stdout_artifact_id="art_stdout",
            stderr_artifact_id="art_stderr",
            status=ValidityStatus.VERIFIED,
            reason="procedure validator exit code matched expected",
        ),
        RLMRankRequest(
            query="parser",
            candidates=(
                RLMRankedCandidate(
                    candidate_id=memory.memory_id,
                    title=memory.title,
                    summary=memory.summary,
                    status=ValidityStatus.VERIFIED,
                ),
            ),
        ),
        RLMRankResponse(
            candidates=(
                RLMRankedCandidate(
                    candidate_id=memory.memory_id,
                    title=memory.title,
                    summary=memory.summary,
                    status=ValidityStatus.VERIFIED,
                    rlm_score=0.8,
                    final_score=0.8,
                    reason="direct match",
                ),
            ),
            model_name="fake-model",
        ),
        BenchmarkManifest(
            manifest_id="synthetic",
            repos=(
                RepoSpec(
                    repo_name="synthetic",
                    source_kind="synthetic_python_smoke",
                    commit_pairs=(CommitPair(pair_id="pair", base_ref="base", head_ref="head"),),
                ),
            ),
            coding_tasks=(
                CodingTaskCase(
                    case_id="task",
                    repo_name="synthetic",
                    commit_pair_id="pair",
                    task_statement="Fix the synthetic task.",
                ),
            ),
        ),
        BenchmarkRunConfig(
            manifest_path="benchmarks/manifests/synthetic-smoke.json",
            suite="retrieval",
            output_dir=".benchmarks/synthetic",
        ),
        EmbeddingIndexEntry(
            memory_id=memory.memory_id,
            adapter_id="deterministic_hash:64",
            content_digest="digest",
            vector=(0.1, 0.2, 0.3),
        ),
        ConsolidationAction(
            action_type="memory_superseded",
            canonical_memory_id=memory.memory_id,
            affected_memory_ids=("mem_old",),
        ),
        ConsolidationRunResult(
            scanned_memory_count=2,
            candidate_group_count=1,
            action_count=1,
            actions=(
                ConsolidationAction(
                    action_type="memory_superseded",
                    canonical_memory_id=memory.memory_id,
                    affected_memory_ids=("mem_old",),
                ),
            ),
        ),
    ]
    cache_key = CacheKey.from_parts("tool", {"b": 2, "a": 1}, repo_fp, env_fp)
    models.append(CacheEntry(key=cache_key, value={"answer": 42}))

    for model in models:
        restored = type(model).from_json(model.to_json())
        assert restored == model


def test_verified_memory_requires_dependency_fingerprint(scope, artifact_evidence) -> None:
    with pytest.raises(ValidationError):
        MemoryItem(
            kind=MemoryKind.CLAIM,
            title="Broken verified memory",
            summary="Missing dependency fingerprint",
            payload=ClaimPayload(claim="Missing dependency fingerprint"),
            evidence=(artifact_evidence,),
            visibility=scope,
            validity=ValidityState(status=ValidityStatus.VERIFIED),
        )


def test_verified_claim_requires_evidence(scope, dep_fp) -> None:
    with pytest.raises(ValidationError):
        MemoryItem(
            kind=MemoryKind.CLAIM,
            title="Broken evidence",
            summary="No evidence attached",
            payload=ClaimPayload(claim="No evidence attached"),
            evidence=(),
            visibility=scope,
            validity=ValidityState(
                status=ValidityStatus.VERIFIED,
                dependency_fingerprint=dep_fp,
            ),
        )


def test_summary_card_requires_memory_or_artifact_evidence(scope, anchor_evidence) -> None:
    with pytest.raises(ValidationError):
        MemoryItem(
            kind=MemoryKind.SUMMARY_CARD,
            title="Broken summary card",
            summary="Summary card with unsupported evidence",
            payload=SummaryCardPayload(summary="Summary card"),
            evidence=(anchor_evidence,),
            visibility=scope,
        )


def test_evidence_kind_enforces_matching_target() -> None:
    with pytest.raises(ValidationError):
        EvidenceRef(kind=EvidenceKind.MEMORY, ref_id="memory:1")


def test_kind_must_match_payload(scope, dep_fp, artifact_evidence) -> None:
    with pytest.raises(ValidationError):
        MemoryItem(
            kind=MemoryKind.DECISION,
            title="Mismatched payload",
            summary="Claim payload under decision kind",
            payload=ClaimPayload(claim="mismatch"),
            evidence=(artifact_evidence,),
            visibility=scope,
            validity=ValidityState(
                status=ValidityStatus.VERIFIED,
                dependency_fingerprint=dep_fp,
            ),
        )
