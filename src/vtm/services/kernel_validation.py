from __future__ import annotations

from vtm.adapters.git import GitRepoFingerprintCollector
from vtm.adapters.runtime import RuntimeEnvFingerprintCollector
from vtm.artifacts import ArtifactRecord
from vtm.base import utc_now
from vtm.enums import ArtifactCaptureState, EvidenceKind, MemoryKind, ValidityStatus
from vtm.events import MemoryEvent
from vtm.evidence import ArtifactRef, EvidenceRef
from vtm.fingerprints import DependencyFingerprint
from vtm.memory_items import LineageEdge, MemoryItem, ProcedurePayload
from vtm.services.fingerprints import DependencyFingerprintBuilder
from vtm.services.kernel_mutations import MetadataMutationRunner
from vtm.services.procedures import ProcedureValidator
from vtm.services.verifier import Verifier
from vtm.stores.base import ArtifactStore, MetadataStore
from vtm.verification import ProcedureValidationResult, VerificationResult

COMMITTED_PROCEDURE_EVIDENCE_STATUSES = {
    ValidityStatus.VERIFIED,
    ValidityStatus.RELOCATED,
}


class ValidationKernelOps:
    def __init__(
        self,
        *,
        metadata_store: MetadataStore,
        artifact_store: ArtifactStore,
        verifier: Verifier,
        procedure_validator: ProcedureValidator,
        mutations: MetadataMutationRunner,
    ) -> None:
        self._metadata_store = metadata_store
        self._artifact_store = artifact_store
        self._verifier = verifier
        self._procedure_validator = procedure_validator
        self._mutations = mutations

    def verify_memory(
        self,
        memory_id: str,
        current_dependency: DependencyFingerprint,
    ) -> tuple[MemoryItem, VerificationResult]:
        item = self._metadata_store.get_memory_item(memory_id)
        if item is None:
            raise KeyError(f"unknown memory item: {memory_id}")
        result = self._verifier.verify(item, current_dependency)
        updated_item = item.model_copy(
            update={
                "validity": result.updated_validity,
                "evidence": result.updated_evidence or item.evidence,
                "updated_at": result.checked_at,
                "stats": item.stats.model_copy(
                    update={
                        "verification_count": item.stats.verification_count + 1,
                        "last_verified_at": result.checked_at,
                    }
                ),
            }
        )
        updated_item = self._mutations.run(
            lambda: self._persist_verified_memory(updated_item),
            build_events=lambda persisted: (
                MemoryEvent(
                    event_type="memory_verified",
                    memory_id=memory_id,
                    payload={"status": result.current_status.value},
                ),
            ),
        )
        return updated_item, result

    def validate_procedure(
        self,
        memory_id: str,
        *,
        repo_root: str | None = None,
    ) -> ProcedureValidationResult:
        item = self._metadata_store.get_memory_item(memory_id)
        if item is None:
            raise KeyError(f"unknown memory item: {memory_id}")
        self._require_procedure_payload(item)

        result = self._procedure_validator.validate(item, repo_root=repo_root)
        dependency_fingerprint = self._procedure_dependency_fingerprint(item, repo_root=repo_root)
        if dependency_fingerprint is None and result.status is ValidityStatus.VERIFIED:
            raise ValueError(
                "successful procedure validation requires repo_root or "
                "an existing dependency fingerprint"
            )

        stdout_record = self._artifact_store.get_artifact_record_by_id(result.stdout_artifact_id)
        stderr_record = self._artifact_store.get_artifact_record_by_id(result.stderr_artifact_id)
        if stdout_record is None or stderr_record is None:
            raise KeyError("procedure validator returned unknown artifact records")
        validation_evidence = (
            self._artifact_evidence(
                stdout_record,
                label="procedure-validator-stdout",
                summary="Procedure validator stdout",
            ),
            self._artifact_evidence(
                stderr_record,
                label="procedure-validator-stderr",
                summary="Procedure validator stderr",
            ),
        )
        merged_evidence = self._merge_evidence(item.evidence, validation_evidence)
        updated_memory = item.model_copy(
            update={
                "evidence": merged_evidence,
                "validity": item.validity.model_copy(
                    update={
                        "status": result.status,
                        "dependency_fingerprint": dependency_fingerprint
                        or item.validity.dependency_fingerprint,
                        "checked_at": result.checked_at,
                        "reason": result.reason,
                    }
                ),
                "stats": item.stats.model_copy(
                    update={
                        "verification_count": item.stats.verification_count + 1,
                        "last_verified_at": result.checked_at,
                    }
                ),
                "metadata": {
                    **item.metadata,
                    "latest_procedure_validation": result.model_dump(mode="json"),
                },
                "updated_at": result.checked_at,
            }
        )
        self._mutations.run(
            lambda: self._persist_validated_procedure(updated_memory),
            build_events=lambda persisted: (
                MemoryEvent(
                    event_type="procedure_validated",
                    memory_id=memory_id,
                    payload=result.model_dump(mode="json"),
                ),
            ),
        )
        self._artifact_store.commit_artifact(result.stdout_artifact_id)
        self._artifact_store.commit_artifact(result.stderr_artifact_id)
        return result

    def promote_to_procedure(
        self,
        source_memory_ids: tuple[str, ...],
        procedure: MemoryItem,
    ) -> MemoryItem:
        if not source_memory_ids:
            raise ValueError("procedure promotion requires at least one source memory")

        payload = self._require_procedure_payload(procedure)
        source_items = tuple(
            self._require_promotable_source(memory_id) for memory_id in source_memory_ids
        )
        source_evidence = tuple(
            EvidenceRef(
                kind=EvidenceKind.MEMORY,
                ref_id=f"memory:{source.memory_id}",
                memory_id=source.memory_id,
                summary="Promoted source memory",
            )
            for source in source_items
        )
        lineage_edges = tuple(
            LineageEdge(
                parent_id=source.memory_id,
                child_id=procedure.memory_id,
                edge_type="promoted_from",
            )
            for source in source_items
        )
        promoted = procedure.model_copy(
            update={
                "evidence": self._merge_evidence(procedure.evidence, source_evidence),
                "lineage": self._merge_lineage(procedure.lineage, lineage_edges),
                "metadata": {
                    **procedure.metadata,
                    "promoted_from_memory_ids": source_memory_ids,
                },
                "updated_at": utc_now(),
            }
        )

        if payload.validator is None and promoted.validity.status is ValidityStatus.PENDING:
            common_dependency = self._common_dependency_fingerprint(source_items)
            if common_dependency is None:
                raise ValueError(
                    "promoting a procedure without a validator requires "
                    "a common dependency fingerprint"
                )
            promoted = promoted.model_copy(
                update={
                    "validity": promoted.validity.model_copy(
                        update={
                            "status": ValidityStatus.VERIFIED,
                            "dependency_fingerprint": common_dependency,
                            "checked_at": utc_now(),
                            "reason": "promoted from verified source memories",
                        }
                    )
                }
            )

        self.validate_committable_item(promoted)
        return self._mutations.run(
            lambda: self._persist_promoted_procedure(promoted, lineage_edges),
            build_events=lambda persisted: (
                MemoryEvent(
                    event_type="procedure_promoted",
                    memory_id=promoted.memory_id,
                    payload={
                        "source_memory_ids": source_memory_ids,
                        "has_validator": payload.validator is not None,
                    },
                ),
            ),
        )

    def validate_committable_item(self, item: MemoryItem) -> None:
        if item.kind is not MemoryKind.PROCEDURE:
            return
        payload = self._require_procedure_payload(item)
        if payload.validator is None and not item.evidence:
            raise ValueError("procedure commits require a validator or evidence")
        if (
            payload.validator is None
            and item.validity.status not in COMMITTED_PROCEDURE_EVIDENCE_STATUSES
        ):
            raise ValueError(
                "procedures without validators must be backed by verified or relocated evidence"
            )

    def _artifact_evidence(
        self,
        record: ArtifactRecord,
        *,
        label: str | None = None,
        summary: str | None = None,
    ) -> EvidenceRef:
        return EvidenceRef(
            kind=EvidenceKind.ARTIFACT,
            ref_id=f"artifact:{record.artifact_id}",
            artifact_ref=ArtifactRef(
                artifact_id=record.artifact_id,
                sha256=record.sha256,
                content_type=record.content_type,
            ),
            label=label,
            summary=summary,
        )

    def _require_artifact_record(self, artifact_id: str) -> ArtifactRecord:
        record = self._artifact_store.get_artifact_record_by_id(artifact_id)
        if record is None:
            raise KeyError(f"unknown artifact record: {artifact_id}")
        if record.capture_state is not ArtifactCaptureState.COMMITTED:
            raise ValueError(f"artifact is not committed: {artifact_id}")
        return record

    def _require_procedure_payload(self, item: MemoryItem) -> ProcedurePayload:
        if item.kind is not MemoryKind.PROCEDURE or not isinstance(item.payload, ProcedurePayload):
            raise ValueError("operation requires a procedure memory item")
        return item.payload

    def _procedure_dependency_fingerprint(
        self,
        item: MemoryItem,
        *,
        repo_root: str | None,
    ) -> DependencyFingerprint | None:
        existing = item.validity.dependency_fingerprint
        repo_candidate = repo_root
        if repo_candidate is None and existing is not None:
            repo_candidate = existing.repo.repo_root
        if repo_candidate is None:
            payload = self._require_procedure_payload(item)
            if payload.validator is not None:
                raw_cwd = payload.validator.config.get("cwd")
                if isinstance(raw_cwd, str):
                    repo_candidate = raw_cwd
        if repo_candidate is None:
            return existing

        builder = DependencyFingerprintBuilder(
            repo_collector=GitRepoFingerprintCollector(),
            env_collector=RuntimeEnvFingerprintCollector(),
        )
        try:
            return builder.build(
                repo_candidate,
                dependency_ids=existing.dependency_ids if existing is not None else (),
                input_digests=existing.input_digests if existing is not None else (),
            )
        except Exception:
            return existing

    def _persist_verified_memory(self, updated_item: MemoryItem) -> MemoryItem:
        self._metadata_store.save_memory_item(updated_item)
        return updated_item

    def _persist_validated_procedure(self, updated_memory: MemoryItem) -> MemoryItem:
        self._metadata_store.save_memory_item(updated_memory)
        return updated_memory

    def _persist_promoted_procedure(
        self,
        promoted: MemoryItem,
        lineage_edges: tuple[LineageEdge, ...],
    ) -> MemoryItem:
        self._metadata_store.save_memory_item(promoted)
        for edge in lineage_edges:
            self._metadata_store.save_lineage_edge(edge)
        return promoted

    def _merge_evidence(
        self,
        existing: tuple[EvidenceRef, ...],
        additions: tuple[EvidenceRef, ...],
    ) -> tuple[EvidenceRef, ...]:
        merged: list[EvidenceRef] = []
        seen: set[tuple[str, str]] = set()
        for evidence in (*existing, *additions):
            key = (evidence.kind.value, evidence.ref_id)
            if key in seen:
                continue
            seen.add(key)
            merged.append(evidence)
        return tuple(merged)

    def _merge_lineage(
        self,
        existing: tuple[LineageEdge, ...],
        additions: tuple[LineageEdge, ...],
    ) -> tuple[LineageEdge, ...]:
        merged: list[LineageEdge] = []
        seen: set[tuple[str, str, str]] = set()
        for edge in (*existing, *additions):
            key = (edge.parent_id, edge.child_id, edge.edge_type)
            if key in seen:
                continue
            seen.add(key)
            merged.append(edge)
        return tuple(merged)

    def _require_promotable_source(self, memory_id: str) -> MemoryItem:
        source = self._metadata_store.get_memory_item(memory_id)
        if source is None:
            raise KeyError(f"unknown source memory item: {memory_id}")
        if source.validity.status not in COMMITTED_PROCEDURE_EVIDENCE_STATUSES:
            raise ValueError(
                f"procedure promotion requires verified or relocated source memories: {memory_id}"
            )
        return source

    def _common_dependency_fingerprint(
        self,
        source_items: tuple[MemoryItem, ...],
    ) -> DependencyFingerprint | None:
        first = source_items[0].validity.dependency_fingerprint
        if first is None:
            return None
        if all(source.validity.dependency_fingerprint == first for source in source_items[1:]):
            return first
        return None
