from __future__ import annotations

from datetime import timedelta

from vtm.enums import DetailLevel, ValidityStatus
from vtm.retrieval import RetrieveRequest
from vtm.services.consolidator import DeterministicConsolidator
from vtm.services.retriever import LexicalRetriever


def test_deterministic_consolidator_supersedes_older_duplicates(
    metadata_store,
    memory_factory,
    scope,
) -> None:
    older = memory_factory(title="Parser result", summary="Parser output stable")
    newer = memory_factory(title=" parser   result ", summary="parser output stable")
    newer = newer.model_copy(update={"updated_at": older.updated_at + timedelta(seconds=5)})
    metadata_store.save_memory_item(older)
    metadata_store.save_memory_item(newer)

    result = DeterministicConsolidator(
        metadata_store=metadata_store,
        event_store=metadata_store,
    ).run()

    persisted_older = metadata_store.get_memory_item(older.memory_id)
    persisted_newer = metadata_store.get_memory_item(newer.memory_id)
    lineage = metadata_store.list_lineage_edges(child_id=newer.memory_id)

    assert persisted_older is not None
    assert persisted_newer is not None
    assert persisted_older.validity.status is ValidityStatus.SUPERSEDED
    assert persisted_newer.validity.status is ValidityStatus.VERIFIED
    assert lineage[0].parent_id == older.memory_id
    assert result.action_count == 1
    assert result.actions[0].canonical_memory_id == newer.memory_id
    assert any(event.event_type == "memory_consolidated" for event in metadata_store.list_events())


def test_deterministic_consolidator_is_idempotent_and_creates_summary_card(
    metadata_store,
    memory_factory,
    scope,
) -> None:
    first = memory_factory(title="Lint result", summary="Linter diagnostics stable")
    second = memory_factory(title="lint result", summary="linter diagnostics stable")
    second = second.model_copy(update={"updated_at": first.updated_at + timedelta(seconds=5)})
    metadata_store.save_memory_item(first)
    metadata_store.save_memory_item(second)

    consolidator = DeterministicConsolidator(
        metadata_store=metadata_store,
        event_store=metadata_store,
        create_summary_cards=True,
    )
    first_run = consolidator.run()
    second_run = consolidator.run()

    summary_cards = [
        memory
        for memory in metadata_store.list_memory_items()
        if memory.kind.value == "summary_card"
    ]

    assert first_run.action_count == 2
    assert second_run.action_count == 0
    assert len(summary_cards) == 1
    assert summary_cards[0].payload.detail_level is DetailLevel.SUMMARY
    assert set(summary_cards[0].payload.supporting_memory_ids) == {
        first.memory_id,
        second.memory_id,
    }
    assert {evidence.memory_id for evidence in summary_cards[0].evidence} == {
        first.memory_id,
        second.memory_id,
    }

    retriever = LexicalRetriever(metadata_store)
    default_result = retriever.retrieve(RetrieveRequest(query="lint", scopes=(scope,)))
    explicit_superseded = retriever.retrieve(
        RetrieveRequest(
            query="lint",
            scopes=(scope,),
            statuses=(ValidityStatus.SUPERSEDED,),
        )
    )

    assert first.memory_id not in {
        candidate.memory.memory_id for candidate in default_result.candidates
    }
    assert explicit_superseded.candidates[0].memory.memory_id == first.memory_id
