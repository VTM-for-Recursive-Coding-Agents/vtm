from __future__ import annotations

from vtm.enums import ValidityStatus
from vtm.memory_items import ValidityState
from vtm.retrieval import RetrieveRequest
from vtm.services.retriever import LexicalRetriever


def test_embedding_retriever_finds_match_with_weak_lexical_overlap(
    metadata_store,
    embedding_retriever,
    memory_factory,
    scope,
) -> None:
    lexical_miss = memory_factory(
        title="Linter memory",
        summary="linter diagnostics stayed stable",
        tags=("tooling",),
    )
    unrelated = memory_factory(
        title="Parser memory",
        summary="parser result stayed stable",
        tags=("parser",),
    )
    metadata_store.save_memory_item(lexical_miss)
    metadata_store.save_memory_item(unrelated)

    lexical = LexicalRetriever(metadata_store).retrieve(
        RetrieveRequest(query="lint", scopes=(scope,))
    )
    embedding = embedding_retriever.retrieve(
        RetrieveRequest(query="lint", scopes=(scope,))
    )

    assert lexical.candidates == ()
    assert embedding.candidates[0].memory.memory_id == lexical_miss.memory_id
    assert embedding.candidates[0].score > 0.0
    assert embedding.candidates[0].explanation.metadata["lexical_score"] == 0.0


def test_embedding_retriever_lazily_indexes_and_refreshes_changed_memory(
    metadata_store,
    embedding_store,
    embedding_retriever,
    memory_factory,
    scope,
) -> None:
    memory = memory_factory(title="Initial title", summary="initial linter output")
    metadata_store.save_memory_item(memory)

    first_result = embedding_retriever.retrieve(
        RetrieveRequest(query="lint", scopes=(scope,))
    )
    first_entry = embedding_store.get_entry(
        memory.memory_id,
        embedding_retriever.adapter_id,
    )

    assert first_result.candidates[0].memory.memory_id == memory.memory_id
    assert first_entry is not None

    updated_memory = memory.model_copy(update={"summary": "updated linter diagnostics"})
    metadata_store.save_memory_item(updated_memory)
    second_result = embedding_retriever.retrieve(
        RetrieveRequest(query="lint", scopes=(scope,))
    )
    second_entry = embedding_store.get_entry(
        memory.memory_id,
        embedding_retriever.adapter_id,
    )

    assert second_result.candidates[0].memory.summary == "updated linter diagnostics"
    assert second_entry is not None
    assert second_entry.content_digest != first_entry.content_digest
    assert second_entry.created_at == first_entry.created_at
    assert second_entry.updated_at >= first_entry.updated_at
    assert len(embedding_store.list_entries()) == 1


def test_embedding_retriever_matches_lexical_status_filters(
    metadata_store,
    embedding_retriever,
    memory_factory,
    scope,
    dep_fp,
) -> None:
    verified = memory_factory(title="Visible linter note", summary="linter diagnostics stable")
    quarantined = memory_factory(
        title="Hidden linter note",
        summary="linter diagnostics hidden",
        validity_status=ValidityStatus.QUARANTINED,
        dependency=dep_fp,
    ).model_copy(
        update={
            "validity": ValidityState(
                status=ValidityStatus.QUARANTINED,
                dependency_fingerprint=dep_fp,
            )
        }
    )
    metadata_store.save_memory_item(verified)
    metadata_store.save_memory_item(quarantined)

    default_result = embedding_retriever.retrieve(
        RetrieveRequest(query="lint", scopes=(scope,))
    )
    explicit_result = embedding_retriever.retrieve(
        RetrieveRequest(
            query="lint",
            scopes=(scope,),
            statuses=(ValidityStatus.QUARANTINED,),
            allow_quarantined=True,
        )
    )

    assert [candidate.memory.memory_id for candidate in default_result.candidates] == [
        verified.memory_id
    ]
    assert [candidate.memory.memory_id for candidate in explicit_result.candidates] == [
        quarantined.memory_id
    ]
