from __future__ import annotations

from pathlib import Path

from vtm.adapters.python_ast import PythonAstSyntaxAdapter
from vtm.enums import EvidenceBudget, ValidityStatus
from vtm.evidence import EvidenceRef
from vtm.memory_items import ValidityState
from vtm.retrieval import RetrieveRequest
from vtm.services.retriever import LexicalRetriever


def _commit_memory(kernel, scope, *items) -> None:
    tx = kernel.begin_transaction(scope)
    for item in items:
        kernel.stage_memory_item(tx.tx_id, item)
    kernel.commit_transaction(tx.tx_id)


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


def test_naive_vs_verified_retrieval_semantics(
    kernel,
    scope,
    memory_factory,
    anchor_evidence,
    dep_fp,
) -> None:
    changed_dep = dep_fp.model_copy(update={"input_digests": ("input-2",)})
    verified = memory_factory(title="Parser verified", summary="parser verified")
    pending = memory_factory(
        title="Parser pending",
        summary="parser pending",
        validity_status=ValidityStatus.PENDING,
        dependency=dep_fp,
    )
    stale = memory_factory(
        title="Parser stale",
        summary="parser stale",
        evidence=(anchor_evidence,),
        dependency=changed_dep,
    )
    _commit_memory(kernel, scope, verified, pending, stale)

    naive = kernel.retrieve(
        RetrieveRequest(
            query="parser",
            scopes=(scope,),
            statuses=tuple(ValidityStatus),
            limit=10,
        )
    )
    verified_only = kernel.retrieve(
        RetrieveRequest(
            query="parser",
            scopes=(scope,),
            statuses=tuple(ValidityStatus),
            limit=10,
            current_dependency=dep_fp,
            verify_on_read=True,
            return_verified_only=True,
        )
    )

    assert {candidate.memory.memory_id for candidate in naive.candidates} == {
        verified.memory_id,
        pending.memory_id,
        stale.memory_id,
    }
    assert {candidate.memory.memory_id for candidate in verified_only.candidates} == {
        verified.memory_id,
    }
    assert verified_only.verified_count == 1
    assert verified_only.relocated_count == 0
    assert verified_only.stale_filtered_count == 2
    assert verified_only.stale_hit_rate > 0.0


def test_verify_on_read_relocates_anchor_and_persists_update(
    kernel,
    scope,
    memory_factory,
    dep_fp,
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "sample.py"
    source_path.write_text(
        "def target() -> int:\n"
        "    return 1\n",
        encoding="utf-8",
    )
    builder = PythonAstSyntaxAdapter()
    anchor = builder.build_anchor(str(source_path), "target")
    evidence = EvidenceRef(
        kind="code_anchor",
        ref_id=f"anchor:{anchor.path}:{anchor.symbol}",
        code_anchor=anchor,
        summary="target anchor",
    )
    memory = memory_factory(
        title="Target parser",
        summary="target parser logic",
        evidence=(evidence,),
        dependency=dep_fp,
    )
    _commit_memory(kernel, scope, memory)

    source_path.write_text(
        "def helper() -> int:\n"
        "    return 0\n\n"
        "def target() -> int:\n"
        "    return 1\n",
        encoding="utf-8",
    )
    changed_dep = dep_fp.model_copy(update={"input_digests": ("input-2",)})
    result = kernel.retrieve(
        RetrieveRequest(
            query="target",
            scopes=(scope,),
            statuses=tuple(ValidityStatus),
            current_dependency=changed_dep,
            verify_on_read=True,
            return_verified_only=True,
        )
    )

    assert len(result.candidates) == 1
    assert result.candidates[0].memory.validity.status is ValidityStatus.RELOCATED
    assert result.relocated_count == 1
    persisted = kernel.expand(memory.memory_id)
    assert persisted[0].code_anchor is not None
    assert persisted[0].code_anchor.start_line == 4


def test_verify_on_read_filters_stale_memory(
    kernel,
    scope,
    memory_factory,
    dep_fp,
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "removed.py"
    source_path.write_text(
        "def target() -> int:\n"
        "    return 1\n",
        encoding="utf-8",
    )
    builder = PythonAstSyntaxAdapter()
    anchor = builder.build_anchor(str(source_path), "target")
    evidence = EvidenceRef(
        kind="code_anchor",
        ref_id=f"anchor:{anchor.path}:{anchor.symbol}",
        code_anchor=anchor,
        summary="target anchor",
    )
    memory = memory_factory(
        title="Parser stale",
        summary="parser stale target",
        evidence=(evidence,),
        dependency=dep_fp,
    )
    _commit_memory(kernel, scope, memory)

    source_path.write_text(
        "def replacement() -> int:\n"
        "    return 2\n",
        encoding="utf-8",
    )
    changed_dep = dep_fp.model_copy(update={"input_digests": ("input-2",)})
    result = kernel.retrieve(
        RetrieveRequest(
            query="parser",
            scopes=(scope,),
            statuses=tuple(ValidityStatus),
            current_dependency=changed_dep,
            verify_on_read=True,
            return_verified_only=True,
        )
    )

    assert result.candidates == ()
    assert result.stale_filtered_count == 1


def test_lexical_retriever_boosts_function_name_and_feedback_signature(
    metadata_store,
    memory_factory,
    scope,
) -> None:
    boosted = memory_factory(
        title="Repair lesson",
        summary="Fix the visible failure for the function.",
        tags=("repair",),
    ).model_copy(
        update={
            "metadata": {
                "function_name": "add",
                "feedback_signature": "expected 5 actual 4 NameError List not defined",
                "memory_role": "repair_lesson",
            }
        }
    )
    generic = memory_factory(
        title="Expected actual note",
        summary="expected 5 actual 4",
        tags=("reference",),
    )
    metadata_store.save_memory_item(boosted)
    metadata_store.save_memory_item(generic)

    retriever = LexicalRetriever(metadata_store)
    result = retriever.retrieve(
        RetrieveRequest(
            query="function add expected 5 actual 4 NameError List not defined",
            scopes=(scope,),
            limit=2,
        )
    )

    assert [candidate.memory.memory_id for candidate in result.candidates] == [
        boosted.memory_id,
        generic.memory_id,
    ]
    assert result.candidates[0].explanation.metadata["lexical_boost_score"] > 0.0
    assert "metadata:function_name" in result.candidates[0].explanation.matched_fields
    assert "metadata:feedback_signature" in result.candidates[0].explanation.matched_fields


def test_lexical_retriever_boosts_anchor_symbol_and_path_matches(
    metadata_store,
    memory_factory,
    scope,
    anchor_evidence,
) -> None:
    anchored = memory_factory(
        title="Builder lesson",
        summary="Use the stored symbol-specific repair guidance.",
        evidence=(anchor_evidence.model_copy(update={"summary": "src/parser.py::target"}),),
        tags=("builder",),
    )
    generic = memory_factory(
        title="Parser note",
        summary="parser target guidance",
        tags=("parser", "target"),
    )
    metadata_store.save_memory_item(anchored)
    metadata_store.save_memory_item(generic)

    retriever = LexicalRetriever(metadata_store)
    result = retriever.retrieve(
        RetrieveRequest(
            query="src example py target",
            scopes=(scope,),
            limit=2,
        )
    )

    assert [candidate.memory.memory_id for candidate in result.candidates] == [
        anchored.memory_id,
        generic.memory_id,
    ]
    assert result.candidates[0].explanation.metadata["lexical_boost_score"] > 0.0
    assert "anchor_path" in result.candidates[0].explanation.matched_fields
    assert "anchor_symbol" in result.candidates[0].explanation.matched_fields
