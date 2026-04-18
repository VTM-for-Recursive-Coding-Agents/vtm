from __future__ import annotations

from pathlib import Path

from vtm.adapters.python_ast import PythonAstSyntaxAdapter
from vtm.enums import EvidenceKind, ValidityStatus
from vtm.evidence import EvidenceRef
from vtm.memory_items import ValidityState
from vtm.services.verifier import BasicVerifier


def test_unchanged_dependencies_skip_verified_memory(memory_factory, dep_fp) -> None:
    verifier = BasicVerifier()
    item = memory_factory(validity_status=ValidityStatus.VERIFIED, dependency=dep_fp)

    result = verifier.verify(item, dep_fp)

    assert result.current_status is ValidityStatus.VERIFIED
    assert result.skipped is True
    assert result.dependency_changed is False


def test_unchanged_dependencies_preserve_stale_memory(memory_factory, dep_fp) -> None:
    verifier = BasicVerifier()
    item = memory_factory(validity_status=ValidityStatus.STALE, dependency=dep_fp)

    result = verifier.verify(item, dep_fp)

    assert result.current_status is ValidityStatus.STALE
    assert result.current_status is not ValidityStatus.VERIFIED
    assert result.skipped is True
    assert result.dependency_changed is False


def test_changed_anchor_dependency_marks_memory_stale(
    memory_factory,
    anchor_evidence,
    dep_fp,
) -> None:
    verifier = BasicVerifier()
    item = memory_factory(evidence=(anchor_evidence,), dependency=dep_fp)
    changed_dependency = dep_fp.model_copy(update={"input_digests": ("input-2",)})

    result = verifier.verify(item, changed_dependency)

    assert result.current_status is ValidityStatus.STALE
    assert result.dependency_changed is True


def test_pending_memory_stays_pending_when_dependencies_match(
    memory_factory,
    dep_fp,
) -> None:
    verifier = BasicVerifier()
    item = memory_factory(
        validity_status=ValidityStatus.PENDING,
        dependency=dep_fp,
    )
    item = item.model_copy(
        update={
            "validity": ValidityState(
                status=ValidityStatus.PENDING,
                dependency_fingerprint=dep_fp,
            )
        }
    )

    result = verifier.verify(item, dep_fp)

    assert result.current_status is ValidityStatus.PENDING
    assert result.skipped is True


def test_changed_dependency_with_removed_symbol_becomes_stale(
    tmp_path: Path,
    memory_factory,
    dep_fp,
) -> None:
    source_path = tmp_path / "sample.py"
    source_path.write_text(
        "def target():\n"
        "    return 1\n",
        encoding="utf-8",
    )
    builder = PythonAstSyntaxAdapter()
    anchor = builder.build_anchor(str(source_path), "target")
    source_path.write_text(
        "def replacement():\n"
        "    return 1\n",
        encoding="utf-8",
    )
    evidence = EvidenceRef(
        kind=EvidenceKind.CODE_ANCHOR,
        ref_id=f"anchor:{anchor.path}:{anchor.symbol}",
        code_anchor=anchor,
        summary="target anchor",
    )
    verifier = BasicVerifier(relocator=builder)
    item = memory_factory(evidence=(evidence,), dependency=dep_fp)
    changed_dependency = dep_fp.model_copy(update={"input_digests": ("input-2",)})

    result = verifier.verify(item, changed_dependency)

    assert result.current_status is ValidityStatus.STALE
