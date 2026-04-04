from __future__ import annotations

from pathlib import Path

from vtm.adapters.python_ast import PythonAstSyntaxAdapter
from vtm.enums import ValidityStatus
from vtm.services.verifier import BasicVerifier


def test_python_anchor_relocation_yields_relocated_status(
    tmp_path: Path,
    memory_factory,
    dep_fp,
) -> None:
    source_path = tmp_path / "example.py"
    source_path.write_text(
        "def helper():\n"
        "    return 1\n\n"
        "def target():\n"
        "    return helper()\n",
        encoding="utf-8",
    )
    builder = PythonAstSyntaxAdapter()
    anchor = builder.build_anchor(str(source_path), "target")
    source_path.write_text(
        "def helper():\n"
        "    return 1\n\n"
        "\n"
        "def target():\n"
        "    return helper()\n",
        encoding="utf-8",
    )

    evidence = kernel_like_evidence(builder, anchor)
    verifier = BasicVerifier(relocator=builder)
    item = memory_factory(evidence=(evidence,), dependency=dep_fp)
    changed_dependency = dep_fp.model_copy(update={"input_digests": ("input-2",)})

    result = verifier.verify(item, changed_dependency)

    assert result.current_status is ValidityStatus.RELOCATED
    assert result.relocation is not None
    assert result.updated_evidence is not None
    assert result.updated_evidence[0].code_anchor is not None
    assert result.updated_evidence[0].code_anchor.start_line == anchor.start_line + 1


def kernel_like_evidence(builder: PythonAstSyntaxAdapter, anchor):
    from vtm.enums import EvidenceKind
    from vtm.evidence import EvidenceRef

    return EvidenceRef(
        kind=EvidenceKind.CODE_ANCHOR,
        ref_id=f"anchor:{anchor.path}:{anchor.symbol}",
        code_anchor=anchor,
        summary="python symbol anchor",
    )
