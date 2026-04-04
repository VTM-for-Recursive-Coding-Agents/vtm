from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from collections.abc import Sequence
from typing import Protocol

from vtm.base import utc_now
from vtm.consolidation import ConsolidationAction, ConsolidationRunResult
from vtm.enums import DetailLevel, EvidenceKind, MemoryKind, ValidityStatus
from vtm.events import MemoryEvent
from vtm.evidence import EvidenceRef
from vtm.memory_items import LineageEdge, MemoryItem, SummaryCardPayload, ValidityState
from vtm.services.kernel_mutations import MetadataMutationRunner
from vtm.stores.base import EventStore, MetadataStore

ACTIVE_CONSOLIDATION_STATUSES = {
    ValidityStatus.VERIFIED,
    ValidityStatus.RELOCATED,
}
WORD_RE = re.compile(r"\s+")


class Consolidator(Protocol):
    def run(self) -> ConsolidationRunResult: ...


class NoopConsolidator:
    def run(self) -> ConsolidationRunResult:
        now = utc_now()
        return ConsolidationRunResult(
            scanned_memory_count=0,
            candidate_group_count=0,
            action_count=0,
            actions=(),
            started_at=now,
            completed_at=now,
        )


class DeterministicConsolidator:
    def __init__(
        self,
        *,
        metadata_store: MetadataStore,
        event_store: EventStore,
        create_summary_cards: bool = False,
    ) -> None:
        self._metadata_store = metadata_store
        self._mutations = MetadataMutationRunner(
            metadata_store=metadata_store,
            event_store=event_store,
        )
        self._create_summary_cards = create_summary_cards

    def run(self) -> ConsolidationRunResult:
        started_at = utc_now()
        memories = tuple(self._metadata_store.list_memory_items())
        groups = self._candidate_groups(memories)
        actions: list[ConsolidationAction] = []

        for group in groups.values():
            group_actions = self._consolidate_group(group, memories)
            actions.extend(group_actions)

        completed_at = utc_now()
        return ConsolidationRunResult(
            scanned_memory_count=len(memories),
            candidate_group_count=len(groups),
            action_count=len(actions),
            actions=tuple(actions),
            started_at=started_at,
            completed_at=completed_at,
        )

    def _candidate_groups(
        self,
        memories: Sequence[MemoryItem],
    ) -> dict[tuple[str, ...], list[MemoryItem]]:
        groups: dict[tuple[str, ...], list[MemoryItem]] = defaultdict(list)
        for memory in memories:
            if memory.kind not in {
                MemoryKind.CLAIM,
                MemoryKind.PROCEDURE,
                MemoryKind.CONSTRAINT,
                MemoryKind.DECISION,
            }:
                continue
            groups[self._group_key(memory)].append(memory)
        return {
            key: group
            for key, group in groups.items()
            if len(group) > 1
            and sum(memory.validity.status in ACTIVE_CONSOLIDATION_STATUSES for memory in group) > 1
        }

    def _consolidate_group(
        self,
        group: Sequence[MemoryItem],
        all_memories: Sequence[MemoryItem],
    ) -> tuple[ConsolidationAction, ...]:
        active = [
            memory for memory in group if memory.validity.status in ACTIVE_CONSOLIDATION_STATUSES
        ]
        if len(active) <= 1:
            return ()
        active.sort(key=lambda item: (-item.updated_at.timestamp(), item.memory_id))
        canonical = active[0]
        duplicates = tuple(active[1:])
        existing_edges = {
            (edge.parent_id, edge.child_id, edge.edge_type)
            for edge in self._metadata_store.list_lineage_edges(child_id=canonical.memory_id)
        }

        persisted_duplicates: list[MemoryItem] = []
        new_edges: list[LineageEdge] = []
        for duplicate in duplicates:
            lineage_edge_key = (
                duplicate.memory_id,
                canonical.memory_id,
                "superseded_by_consolidation",
            )
            lineage_edge = LineageEdge(
                parent_id=duplicate.memory_id,
                child_id=canonical.memory_id,
                edge_type="superseded_by_consolidation",
            )
            if lineage_edge_key not in existing_edges:
                new_edges.append(lineage_edge)
                existing_edges.add(lineage_edge_key)
            persisted_duplicates.append(
                self._supersede_memory(
                    duplicate,
                    canonical_memory_id=canonical.memory_id,
                    lineage_edge=lineage_edge,
                )
            )

        actions: list[ConsolidationAction] = []
        if persisted_duplicates or new_edges:
            self._mutations.run(
                lambda: self._persist_group_changes(persisted_duplicates, new_edges),
                build_events=lambda _result: tuple(
                    MemoryEvent(
                        event_type="memory_consolidated",
                        memory_id=duplicate.memory_id,
                        payload={
                            "canonical_memory_id": canonical.memory_id,
                            "action": "memory_superseded",
                        },
                    )
                    for duplicate in duplicates
                ),
            )
            actions.extend(
                ConsolidationAction(
                    action_type="memory_superseded",
                    canonical_memory_id=canonical.memory_id,
                    affected_memory_ids=(duplicate.memory_id,),
                    metadata={"group_size": len(group)},
                )
                for duplicate in duplicates
            )

        if self._create_summary_cards:
            summary_card = self._existing_summary_card(group, all_memories)
            if summary_card is None:
                created_card = self._build_summary_card(canonical, group)
                self._mutations.run(
                    lambda: self._metadata_store.save_memory_item(created_card),
                    build_events=lambda _result: (
                        MemoryEvent(
                            event_type="summary_card_created",
                            memory_id=created_card.memory_id,
                            payload={
                                "canonical_memory_id": canonical.memory_id,
                                "supporting_memory_ids": [
                                    memory.memory_id for memory in group
                                ],
                            },
                        ),
                    ),
                )
                actions.append(
                    ConsolidationAction(
                        action_type="summary_card_created",
                        canonical_memory_id=canonical.memory_id,
                        affected_memory_ids=tuple(memory.memory_id for memory in group),
                        created_memory_id=created_card.memory_id,
                        metadata={"group_digest": self._summary_group_digest(group)},
                    )
                )

        return tuple(actions)

    def _persist_group_changes(
        self,
        duplicates: Sequence[MemoryItem],
        lineage_edges: Sequence[LineageEdge],
    ) -> None:
        for duplicate in duplicates:
            self._metadata_store.save_memory_item(duplicate)
        for lineage_edge in lineage_edges:
            self._metadata_store.save_lineage_edge(lineage_edge)

    def _supersede_memory(
        self,
        memory: MemoryItem,
        *,
        canonical_memory_id: str,
        lineage_edge: LineageEdge,
    ) -> MemoryItem:
        existing_lineage = {
            (edge.parent_id, edge.child_id, edge.edge_type): edge for edge in memory.lineage
        }
        existing_lineage.setdefault(
            (lineage_edge.parent_id, lineage_edge.child_id, lineage_edge.edge_type),
            lineage_edge,
        )
        return memory.model_copy(
            update={
                "validity": memory.validity.model_copy(
                    update={
                        "status": ValidityStatus.SUPERSEDED,
                        "checked_at": utc_now(),
                        "reason": (
                            "superseded by deterministic consolidator: "
                            f"{canonical_memory_id}"
                        ),
                    }
                ),
                "lineage": tuple(existing_lineage.values()),
                "updated_at": utc_now(),
            }
        )

    def _existing_summary_card(
        self,
        group: Sequence[MemoryItem],
        memories: Sequence[MemoryItem],
    ) -> MemoryItem | None:
        group_digest = self._summary_group_digest(group)
        for memory in memories:
            if memory.kind is not MemoryKind.SUMMARY_CARD:
                continue
            if memory.metadata.get("consolidation_group_digest") == group_digest:
                return memory
        return None

    def _build_summary_card(
        self,
        canonical: MemoryItem,
        group: Sequence[MemoryItem],
    ) -> MemoryItem:
        supporting_ids = tuple(memory.memory_id for memory in group)
        evidence = tuple(
            EvidenceRef(
                kind=EvidenceKind.MEMORY,
                ref_id=f"memory:{memory.memory_id}",
                memory_id=memory.memory_id,
                summary="Consolidation supporting memory",
            )
            for memory in group
        )
        title = f"Consolidated summary: {canonical.title}"
        summary = (
            f"Canonical memory {canonical.memory_id} summarizes {len(group)} related memories "
            f"in {canonical.visibility.kind.value}:{canonical.visibility.scope_id}."
        )
        return MemoryItem(
            kind=MemoryKind.SUMMARY_CARD,
            title=title,
            summary=summary,
            payload=SummaryCardPayload(
                summary=summary,
                detail_level=DetailLevel.SUMMARY,
                supporting_memory_ids=supporting_ids,
            ),
            evidence=evidence,
            tags=canonical.tags,
            visibility=canonical.visibility,
            validity=ValidityState(
                status=ValidityStatus.VERIFIED,
                dependency_fingerprint=canonical.validity.dependency_fingerprint,
                checked_at=utc_now(),
                reason="generated by deterministic consolidator",
            ),
            metadata={
                "consolidation_group_digest": self._summary_group_digest(group),
                "canonical_memory_id": canonical.memory_id,
                "generated_by": "deterministic_consolidator",
            },
        )

    def _group_key(self, memory: MemoryItem) -> tuple[str, ...]:
        dependency_fingerprint = memory.validity.dependency_fingerprint
        dependency_digest = (
            hashlib.sha256(dependency_fingerprint.to_json().encode("utf-8")).hexdigest()
            if dependency_fingerprint is not None
            else "none"
        )
        return (
            memory.kind.value,
            memory.visibility.kind.value,
            memory.visibility.scope_id,
            self._normalize_text(memory.title),
            self._normalize_text(memory.summary),
            ",".join(sorted(tag.lower() for tag in memory.tags)),
            dependency_digest,
        )

    def _summary_group_digest(self, group: Sequence[MemoryItem]) -> str:
        payload = "|".join(sorted(memory.memory_id for memory in group))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _normalize_text(self, value: str) -> str:
        return WORD_RE.sub(" ", value.strip().lower())
