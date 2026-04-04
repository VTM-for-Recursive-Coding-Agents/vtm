from __future__ import annotations

from typing import Protocol

from vtm.anchors import AnchorRelocation, AnchorRelocator
from vtm.base import utc_now
from vtm.enums import EvidenceKind, ValidityStatus
from vtm.evidence import EvidenceRef
from vtm.fingerprints import DependencyFingerprint
from vtm.memory_items import MemoryItem, ValidityState
from vtm.verification import VerificationResult


class Verifier(Protocol):
    def verify(
        self,
        item: MemoryItem,
        current_dependency: DependencyFingerprint,
    ) -> VerificationResult: ...


class BasicVerifier:
    def __init__(self, *, relocator: AnchorRelocator | None = None) -> None:
        self._relocator = relocator

    def verify(
        self,
        item: MemoryItem,
        current_dependency: DependencyFingerprint,
    ) -> VerificationResult:
        checked_at = utc_now()
        previous_status = item.validity.status
        stored_dependency = item.validity.dependency_fingerprint
        dependency_changed = stored_dependency != current_dependency
        reasons: list[str] = []
        relocation: AnchorRelocation | None = None
        updated_evidence: tuple[EvidenceRef, ...] | None = None
        skipped = False

        if stored_dependency is None:
            current_status = ValidityStatus.UNKNOWN
            reasons.append("missing dependency fingerprint")
        elif not dependency_changed:
            if previous_status in {ValidityStatus.VERIFIED, ValidityStatus.RELOCATED}:
                current_status = previous_status
                skipped = True
                reasons.append("dependency fingerprint unchanged")
            else:
                current_status = ValidityStatus.VERIFIED
                reasons.append("dependency fingerprint unchanged; promoting to verified")
        else:
            anchor_indexes = [
                index
                for index, evidence in enumerate(item.evidence)
                if evidence.kind is EvidenceKind.CODE_ANCHOR and evidence.code_anchor is not None
            ]
            if anchor_indexes and self._relocator is not None:
                relocations: list[tuple[int, AnchorRelocation]] = []
                stale_anchor = False
                for anchor_index in anchor_indexes:
                    anchor = item.evidence[anchor_index].code_anchor
                    assert anchor is not None
                    anchor_relocation = self._relocator.relocate(anchor)
                    if anchor_relocation is None:
                        stale_anchor = True
                        continue
                    relocations.append((anchor_index, anchor_relocation))
                if stale_anchor and not relocations:
                    current_status = ValidityStatus.STALE
                    reasons.append(
                        "dependency fingerprint changed; code anchor could not be relocated"
                    )
                elif relocations:
                    evidence_list = list(item.evidence)
                    relocation = relocations[0][1]
                    spans_changed = False
                    for anchor_index, anchor_relocation in relocations:
                        spans_changed = spans_changed or (
                            anchor_relocation.old_anchor.start_line
                            != anchor_relocation.new_anchor.start_line
                            or anchor_relocation.old_anchor.end_line
                            != anchor_relocation.new_anchor.end_line
                            or anchor_relocation.old_anchor.start_byte
                            != anchor_relocation.new_anchor.start_byte
                            or anchor_relocation.old_anchor.end_byte
                            != anchor_relocation.new_anchor.end_byte
                        )
                        evidence_list[anchor_index] = evidence_list[anchor_index].model_copy(
                            update={"code_anchor": anchor_relocation.new_anchor}
                        )
                    updated_evidence = tuple(evidence_list)
                    current_status = (
                        ValidityStatus.RELOCATED if spans_changed else ValidityStatus.VERIFIED
                    )
                    reasons.append(
                        "dependency fingerprint changed; anchor relocated"
                        if spans_changed
                        else "dependency fingerprint changed; anchor revalidated"
                    )
            elif anchor_indexes:
                current_status = ValidityStatus.STALE
                reasons.append("dependency fingerprint changed; code anchor requires relocation")
            else:
                current_status = ValidityStatus.UNKNOWN
                reasons.append("dependency fingerprint changed")

        updated_validity = ValidityState(
            status=current_status,
            dependency_fingerprint=current_dependency,
            checked_at=checked_at,
            reason=reasons[0] if reasons else None,
        )
        return VerificationResult(
            memory_id=item.memory_id,
            previous_status=previous_status,
            current_status=current_status,
            dependency_changed=dependency_changed,
            checked_at=checked_at,
            reasons=tuple(reasons),
            updated_validity=updated_validity,
            relocation=relocation,
            updated_evidence=updated_evidence,
            skipped=skipped,
        )
