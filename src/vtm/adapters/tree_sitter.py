"""Tree-sitter based syntax anchor construction and relocation."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Protocol

from vtm.anchors import AnchorAdapter, AnchorRelocation, CodeAnchor


class SyntaxTreeAdapter(Protocol):
    """Protocol for syntax-tree based anchor builders."""

    def build_anchor(self, source_path: str, symbol: str) -> CodeAnchor: ...


class SyntaxAnchorAdapter(AnchorAdapter, Protocol):
    """Combined syntax build-and-relocate adapter contract."""

    pass


def _context_digest(source: str, start_line: int, end_line: int) -> str:
    """Hash a small source window around an anchored symbol."""
    lines = source.splitlines()
    start_index = max(start_line - 2, 0)
    end_index = min(end_line + 1, len(lines))
    context = "\n".join(lines[start_index:end_index])
    return hashlib.sha256(context.encode("utf-8")).hexdigest()


class UnavailableTreeSitterAdapter:
    """Fallback adapter used when tree-sitter is unavailable."""

    def build_anchor(self, source_path: str, symbol: str) -> CodeAnchor:
        raise NotImplementedError("Tree-sitter integration is deferred in the kernel scaffold")

    def relocate(self, anchor: CodeAnchor) -> AnchorRelocation | None:
        return None


class PythonTreeSitterSyntaxAdapter:
    """Python anchor adapter backed by tree-sitter with optional fallback."""

    language = "python"

    def __init__(
        self,
        *,
        fallback: AnchorAdapter | None = None,
    ) -> None:
        """Initialize tree-sitter bindings if available."""
        self._fallback = fallback
        self._import_error: ImportError | None = None
        try:
            import tree_sitter_python as tspython
            from tree_sitter import Language, Parser
        except ImportError as exc:
            self._language = None
            self._parser = None
            self._import_error = exc
            return

        self._language = Language(tspython.language())
        self._parser = Parser(self._language)

    def build_anchor(self, source_path: str, symbol: str) -> CodeAnchor:
        """Build a code anchor for a Python symbol."""
        if self._parser is None:
            if self._fallback is None:
                raise NotImplementedError(
                    "Tree-sitter Python support requires the tree-sitter and "
                    "tree-sitter-python packages"
                ) from self._import_error
            return self._fallback.build_anchor(source_path, symbol)

        path = Path(source_path)
        source = path.read_text(encoding="utf-8")
        tree = self._parser.parse(source.encode("utf-8"))
        match = self._find_symbol(tree.root_node, symbol)
        if match is None:
            if self._fallback is None:
                raise KeyError(f"unable to resolve symbol {symbol!r} in {source_path}")
            return self._fallback.build_anchor(source_path, symbol)

        kind, node = match
        start_line = node.start_point[0] + 1
        snippet = source.encode("utf-8")[node.start_byte : node.end_byte].decode("utf-8")
        line_count = snippet.count("\n")
        end_line = start_line + line_count
        if snippet.endswith("\n") and line_count > 0:
            end_line -= 1
        symbol_digest = hashlib.sha256(snippet.encode("utf-8")).hexdigest()
        return CodeAnchor(
            path=str(path),
            symbol=symbol,
            kind=kind,
            language=self.language,
            symbol_digest=symbol_digest,
            context_digest=_context_digest(source, start_line, end_line),
            start_line=start_line,
            end_line=end_line,
            start_byte=node.start_byte,
            end_byte=node.end_byte,
        )

    def relocate(self, anchor: CodeAnchor) -> AnchorRelocation | None:
        """Relocate an existing anchor by rebuilding the same symbol."""
        if anchor.language != self.language or anchor.symbol is None:
            if self._fallback is not None:
                return self._fallback.relocate(anchor)
            return None
        try:
            new_anchor = self.build_anchor(anchor.path, anchor.symbol)
        except (FileNotFoundError, KeyError, UnicodeDecodeError, NotImplementedError):
            if self._fallback is not None:
                return self._fallback.relocate(anchor)
            return None

        if new_anchor.symbol_digest != anchor.symbol_digest:
            if self._fallback is not None:
                return self._fallback.relocate(anchor)
            return None

        if self._same_span(anchor, new_anchor):
            return AnchorRelocation(
                old_anchor=anchor,
                new_anchor=new_anchor,
                method="python_tree_sitter_exact",
                confidence=1.0,
            )

        return AnchorRelocation(
            old_anchor=anchor,
            new_anchor=new_anchor,
            method="python_tree_sitter_relocated",
            confidence=0.98,
        )

    def _find_symbol(self, root_node: Any, target_symbol: str) -> tuple[str, Any] | None:
        stack: list[str] = []
        for kind, node, qualname in self._iter_symbol_nodes(root_node, stack):
            if qualname == target_symbol:
                return kind, node
        return None

    def _iter_symbol_nodes(
        self,
        node: Any,
        stack: list[str],
    ) -> tuple[tuple[str, Any, str], ...]:
        results: list[tuple[str, Any, str]] = []
        name_node = node.child_by_field_name("name")
        if name_node is not None:
            symbol_kind = self._kind_for_node(node)
            if symbol_kind is not None:
                name = name_node.text.decode("utf-8")
                stack.append(name)
                qualname = ".".join(stack)
                results.append((symbol_kind, node, qualname))
                for child in node.named_children:
                    results.extend(self._iter_symbol_nodes(child, stack))
                stack.pop()
                return tuple(results)

        for child in node.named_children:
            results.extend(self._iter_symbol_nodes(child, stack))
        return tuple(results)

    def _kind_for_node(self, node: Any) -> str | None:
        if node.type == "class_definition":
            return "class"
        if node.type == "function_definition":
            source = node.text.decode("utf-8", errors="ignore").lstrip()
            if source.startswith("async def "):
                return "async_function"
            return "function"
        return None

    def _same_span(self, old: CodeAnchor, new: CodeAnchor) -> bool:
        return (
            old.start_line == new.start_line
            and old.end_line == new.end_line
            and old.start_byte == new.start_byte
            and old.end_byte == new.end_byte
        )
