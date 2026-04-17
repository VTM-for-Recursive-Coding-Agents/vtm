from __future__ import annotations

import importlib
import tomllib
from pathlib import Path

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
    CommandValidatorConfig,
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


def test_package_root_exports_core_implementations() -> None:
    from vtm import (
        FilesystemArtifactStore,
        LexicalRetriever,
        SqliteMetadataStore,
        TransactionalMemoryKernel,
    )

    assert TransactionalMemoryKernel.__name__ == "TransactionalMemoryKernel"
    assert LexicalRetriever.__name__ == "LexicalRetriever"
    assert SqliteMetadataStore.__name__ == "SqliteMetadataStore"
    assert FilesystemArtifactStore.__name__ == "FilesystemArtifactStore"


def test_validator_spec_coerces_command_config_through_typed_accessor() -> None:
    validator = ValidatorSpec(
        name="parser-check",
        kind="command",
        config={"command": ["python3", "-c", "print('ok')"], "timeout_seconds": 1},
    )

    config = validator.command_config()

    assert isinstance(config, CommandValidatorConfig)
    assert config.command == ("python3", "-c", "print('ok')")
    assert config.timeout_seconds == 1.0


def test_validator_spec_accepts_prebuilt_command_config() -> None:
    config = CommandValidatorConfig(command=("python3", "-c", "print('ok')"))
    validator = ValidatorSpec(name="parser-check", kind="command", config=config)

    assert validator.command_config() == config


def test_project_scripts_expose_supported_cli_entrypoints() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    scripts = project["project"]["scripts"]
    assert scripts["vtm-bench"] == "vtm.benchmarks.run:main"
    assert (
        scripts["vtm-prepare-swebench-lite"]
        == "vtm.benchmarks.prepare_swebench_lite:main"
    )


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


def test_benchmark_run_config_validates_attempt_controls() -> None:
    with pytest.raises(ValidationError):
        BenchmarkRunConfig(
            manifest_path="benchmarks/manifests/synthetic-smoke.json",
            suite="coding",
            output_dir=".benchmarks/synthetic",
            attempt_count=2,
            pass_k_values=(1, 3),
        )

    with pytest.raises(ValidationError):
        BenchmarkRunConfig(
            manifest_path="benchmarks/manifests/synthetic-smoke.json",
            suite="retrieval",
            output_dir=".benchmarks/synthetic",
            attempt_count=2,
        )

    config = BenchmarkRunConfig(
        manifest_path="benchmarks/manifests/synthetic-smoke.json",
        suite="coding",
        output_dir=".benchmarks/synthetic",
        attempt_count=3,
        pass_k_values=(3, 1, 2),
    )

    assert config.pass_k_values == (3, 1, 2)

    with pytest.raises(ValidationError):
        BenchmarkRunConfig(
            manifest_path="benchmarks/manifests/synthetic-smoke.json",
            suite="drift",
            output_dir=".benchmarks/synthetic",
            seed_on_base_query_on_head=True,
        )


def test_benchmark_run_config_validates_docker_workspace_settings() -> None:
    with pytest.raises(ValidationError):
        BenchmarkRunConfig(
            manifest_path="benchmarks/manifests/synthetic-smoke.json",
            suite="coding",
            output_dir=".benchmarks/synthetic",
            workspace_backend="docker_workspace",
        )

    with pytest.raises(ValidationError):
        BenchmarkRunConfig(
            manifest_path="benchmarks/manifests/synthetic-smoke.json",
            suite="coding",
            output_dir=".benchmarks/synthetic",
            workspace_backend="local_workspace",
            docker_image="python:3.12",
        )

    config = BenchmarkRunConfig(
        manifest_path="benchmarks/manifests/synthetic-smoke.json",
        suite="coding",
        output_dir=".benchmarks/synthetic",
        workspace_backend="docker_workspace",
        docker_image="python:3.12",
        docker_binary="/tmp/fake-docker",
        docker_network="bridge",
    )

    assert config.workspace_backend == "docker_workspace"
    assert config.docker_image == "python:3.12"
    assert config.docker_binary == "/tmp/fake-docker"
    assert config.docker_network == "bridge"


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
