from __future__ import annotations

from pathlib import Path

from vtm.adapters.python_ast import PythonAstSyntaxAdapter
from vtm.adapters.tree_sitter import PythonTreeSitterSyntaxAdapter


def test_python_anchor_builder_supports_function_class_and_method(tmp_path: Path) -> None:
    source_path = tmp_path / "sample.py"
    source_path.write_text(
        "def top_level():\n"
        "    return 1\n\n"
        "class Example:\n"
        "    def method(self):\n"
        "        return top_level()\n",
        encoding="utf-8",
    )
    builder = PythonAstSyntaxAdapter()

    function_anchor = builder.build_anchor(str(source_path), "top_level")
    class_anchor = builder.build_anchor(str(source_path), "Example")
    method_anchor = builder.build_anchor(str(source_path), "Example.method")

    assert function_anchor.kind == "function"
    assert function_anchor.language == "python"
    assert class_anchor.kind == "class"
    assert method_anchor.kind == "function"
    assert method_anchor.symbol == "Example.method"
    assert function_anchor.symbol_digest is not None
    assert method_anchor.context_digest is not None


def test_python_anchor_relocator_returns_none_when_ast_changes(tmp_path: Path) -> None:
    source_path = tmp_path / "sample.py"
    source_path.write_text(
        "def target():\n"
        "    return 1\n",
        encoding="utf-8",
    )
    builder = PythonAstSyntaxAdapter()
    relocator = builder
    original = builder.build_anchor(str(source_path), "target")

    source_path.write_text(
        "def target():\n"
        "    return 2\n",
        encoding="utf-8",
    )

    assert relocator.relocate(original) is None


def test_tree_sitter_python_adapter_matches_ast_builder_for_supported_symbols(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "sample.py"
    source_path.write_text(
        "async def top_level():\n"
        "    return 1\n\n"
        "class Example:\n"
        "    async def method(self):\n"
        "        return await top_level()\n",
        encoding="utf-8",
    )
    fallback = PythonAstSyntaxAdapter()
    adapter = PythonTreeSitterSyntaxAdapter(fallback=fallback)

    for symbol in ("top_level", "Example", "Example.method"):
        tree_sitter_anchor = adapter.build_anchor(str(source_path), symbol)
        ast_anchor = fallback.build_anchor(str(source_path), symbol)
        assert tree_sitter_anchor.symbol == ast_anchor.symbol
        assert tree_sitter_anchor.kind == ast_anchor.kind
        assert tree_sitter_anchor.language == ast_anchor.language
        assert tree_sitter_anchor.start_line == ast_anchor.start_line
        assert tree_sitter_anchor.end_line == ast_anchor.end_line


def test_tree_sitter_python_adapter_relocates_whitespace_only_change(tmp_path: Path) -> None:
    source_path = tmp_path / "sample.py"
    source_path.write_text(
        "def target():\n"
        "    return 1\n",
        encoding="utf-8",
    )
    adapter = PythonTreeSitterSyntaxAdapter(fallback=PythonAstSyntaxAdapter())
    original = adapter.build_anchor(str(source_path), "target")

    source_path.write_text(
        "\n"
        "def target():\n"
        "    return 1\n",
        encoding="utf-8",
    )

    relocation = adapter.relocate(original)
    assert relocation is not None
    assert relocation.new_anchor.start_line == original.start_line + 1
