"""Python AST-based anchor construction and relocation."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

from vtm.anchors import AnchorRelocation, CodeAnchor


def _source_offsets(source: str) -> list[int]:
    offsets = [0]
    running = 0
    for line in source.splitlines(keepends=True):
        running += len(line.encode("utf-8"))
        offsets.append(running)
    return offsets


def _line_start(offsets: list[int], lineno: int) -> int:
    return offsets[max(lineno - 1, 0)]


def _byte_offset(offsets: list[int], lineno: int | None, col_offset: int | None) -> int | None:
    if lineno is None or col_offset is None:
        return None
    return _line_start(offsets, lineno) + col_offset


def _node_digest(node: ast.AST) -> str:
    serialized = ast.dump(node, annotate_fields=True, include_attributes=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _context_digest(source: str, start_line: int, end_line: int) -> str:
    lines = source.splitlines()
    start_index = max(start_line - 2, 0)
    end_index = min(end_line + 1, len(lines))
    context = "\n".join(lines[start_index:end_index])
    return hashlib.sha256(context.encode("utf-8")).hexdigest()


class _QualifiedSymbolFinder(ast.NodeVisitor):
    def __init__(self, target_symbol: str) -> None:
        self._target_symbol = target_symbol
        self._stack: list[str] = []
        self.match: ast.AST | None = None

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._visit_named_node(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_named_node(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_named_node(node)

    def _visit_named_node(
        self,
        node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> None:
        name = node.name
        self._stack.append(name)
        qualname = ".".join(self._stack)
        if qualname == self._target_symbol:
            self.match = node
            self._stack.pop()
            return
        self.generic_visit(node)
        self._stack.pop()


class PythonAstSyntaxAdapter:
    """Pure-Python anchor adapter based on the builtin `ast` module."""

    language = "python"

    def build_anchor(self, source_path: str, symbol: str) -> CodeAnchor:
        """Build a code anchor for a qualified Python symbol."""
        path = Path(source_path)
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        finder = _QualifiedSymbolFinder(symbol)
        finder.visit(tree)
        if finder.match is None:
            raise KeyError(f"unable to resolve symbol {symbol!r} in {source_path}")

        node = self._validated_match(finder.match)
        offsets = _source_offsets(source)
        start_line = node.lineno
        end_line = node.end_lineno or start_line
        start_byte = _byte_offset(offsets, node.lineno, node.col_offset)
        end_byte = _byte_offset(
            offsets,
            node.end_lineno,
            node.end_col_offset,
        )
        kind = self._kind_for_node(node)
        return CodeAnchor(
            path=str(path),
            symbol=symbol,
            kind=kind,
            language=self.language,
            symbol_digest=_node_digest(node),
            context_digest=_context_digest(source, start_line, end_line),
            start_line=start_line,
            end_line=end_line,
            start_byte=start_byte,
            end_byte=end_byte,
        )

    def relocate(self, anchor: CodeAnchor) -> AnchorRelocation | None:
        """Relocate an anchor by rebuilding the same qualified symbol."""
        if anchor.language != "python" or anchor.symbol is None:
            return None
        try:
            new_anchor = self.build_anchor(anchor.path, anchor.symbol)
        except (FileNotFoundError, KeyError, SyntaxError, UnicodeDecodeError):
            return None

        if new_anchor.symbol_digest != anchor.symbol_digest:
            return None

        if self._same_span(anchor, new_anchor):
            return AnchorRelocation(
                old_anchor=anchor,
                new_anchor=new_anchor,
                method="python_ast_qualname_exact",
                confidence=1.0,
            )

        return AnchorRelocation(
            old_anchor=anchor,
            new_anchor=new_anchor,
            method="python_ast_qualname_relocated",
            confidence=0.95,
        )

    def _kind_for_node(self, node: ast.AST) -> str:
        if isinstance(node, ast.ClassDef):
            return "class"
        if isinstance(node, ast.AsyncFunctionDef):
            return "async_function"
        if isinstance(node, ast.FunctionDef):
            return "function"
        raise TypeError(f"unsupported node type for anchor: {type(node)!r}")

    def _validated_match(
        self,
        node: ast.AST,
    ) -> ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            return node
        raise TypeError(f"unsupported node type for anchor: {type(node)!r}")

    def _same_span(self, old: CodeAnchor, new: CodeAnchor) -> bool:
        return (
            old.start_line == new.start_line
            and old.end_line == new.end_line
            and old.start_byte == new.start_byte
            and old.end_byte == new.end_byte
        )


class PythonAstAnchorAdapter(PythonAstSyntaxAdapter):
    """Compatibility alias for the AST anchor builder."""

    pass


class PythonAstAnchorRelocator:
    """Compatibility wrapper exposing only relocation behavior."""

    def __init__(self, *, builder: PythonAstSyntaxAdapter | None = None) -> None:
        """Create a relocator backed by a Python AST anchor builder."""
        self._builder = builder or PythonAstSyntaxAdapter()

    def relocate(self, anchor: CodeAnchor) -> AnchorRelocation | None:
        """Relocate the provided code anchor using the configured builder."""
        return self._builder.relocate(anchor)
