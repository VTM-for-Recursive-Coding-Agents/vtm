from __future__ import annotations

from vtm.enums import EvidenceBudget, ValidityStatus
from vtm.memory_items import ValidityState
from vtm.retrieval import RetrieveRequest
from vtm.services.retriever import LexicalRetriever


def test_default_retrieval_filters_invalid_statuses(
    metadata_store,
    memory_factory,
    scope,
    dep_fp,
) -> None:
    verified = memory_factory(title="Parser verified", summary="parser result stable")
    relocated = memory_factory(
        title="Parser relocated",
        summary="parser anchor moved",
        validity_status=ValidityStatus.RELOCATED,
        dependency=dep_fp,
    )
    stale = memory_factory(
        title="Parser stale",
        summary="parser result stale",
        validity_status=ValidityStatus.STALE,
        dependency=dep_fp,
    )
    quarantined = memory_factory(
        title="Parser quarantined",
        summary="parser issue hidden",
        validity_status=ValidityStatus.QUARANTINED,
        dependency=dep_fp,
    )
    quarantined = quarantined.model_copy(
        update={
            "validity": ValidityState(
                status=ValidityStatus.QUARANTINED,
                dependency_fingerprint=dep_fp,
            )
        }
    )

    for item in (verified, relocated, stale, quarantined):
        metadata_store.save_memory_item(item)

    retriever = LexicalRetriever(metadata_store)
    result = retriever.retrieve(RetrieveRequest(query="parser", scopes=(scope,)))

    assert {candidate.memory.memory_id for candidate in result.candidates} == {
        relocated.memory_id,
        verified.memory_id,
    }
    assert result.total_candidates == 2
    assert all(
        candidate.explanation.matched_tokens == ("parser",)
        for candidate in result.candidates
    )


def test_evidence_budget_controls_raw_evidence(metadata_store, memory_factory, scope) -> None:
    item = memory_factory(title="Budget test", summary="budget parser result")
    metadata_store.save_memory_item(item)
    retriever = LexicalRetriever(metadata_store)

    summary_only = retriever.retrieve(
        RetrieveRequest(
            query="budget",
            scopes=(scope,),
            evidence_budget=EvidenceBudget.SUMMARY_ONLY,
        )
    )
    summary_first = retriever.retrieve(
        RetrieveRequest(
            query="budget",
            scopes=(scope,),
            evidence_budget=EvidenceBudget.SUMMARY_FIRST,
        )
    )
    force_raw = retriever.retrieve(
        RetrieveRequest(
            query="budget",
            scopes=(scope,),
            evidence_budget=EvidenceBudget.FORCE_RAW,
        )
    )

    assert summary_only.candidates[0].evidence == ()
    assert summary_only.candidates[0].raw_evidence_available is False
    assert summary_first.candidates[0].evidence == ()
    assert summary_first.candidates[0].raw_evidence_available is True
    assert force_raw.candidates[0].evidence == item.evidence
    assert force_raw.candidates[0].raw_evidence_available is True
    assert retriever.expand(item.memory_id) == item.evidence
