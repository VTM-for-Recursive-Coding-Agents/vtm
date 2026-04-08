"""Source symbol extraction and case-generation helpers for benchmarks."""

from __future__ import annotations

import ast
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from vtm.benchmarks.models import CommitPair, DriftCase, RetrievalCase
from vtm.enums import ValidityStatus

SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
WORD_RE = re.compile(r"[A-Za-z0-9]+")
STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "into",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "while",
        "with",
    }
)


@dataclass(frozen=True)
class SymbolSnapshot:
    """Normalized summary of a Python symbol extracted from source."""

    relative_path: str
    qualname: str
    kind: str
    start_line: int
    end_line: int
    symbol_digest: str
    summary: str
    query: str
    snippet: str

    @property
    def key(self) -> tuple[str, str]:
        """Return the unique key used for matching symbol snapshots."""
        return (self.relative_path, self.qualname)


class SymbolIndexer:
    """Extracts Python symbols and converts them into benchmark cases."""

    def extract_symbols(self, repo_root: Path) -> dict[tuple[str, str], SymbolSnapshot]:
        """Index all non-test Python symbols in a repository tree."""
        snapshots: dict[tuple[str, str], SymbolSnapshot] = {}
        for source_path in sorted(repo_root.rglob("*.py")):
            if ".git" in source_path.parts:
                continue
            relative_path = source_path.relative_to(repo_root).as_posix()
            if self._is_test_path(relative_path):
                continue
            source = source_path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(source_path))
            for snapshot in self._extract_file_symbols(relative_path, source, tree.body):
                snapshots[snapshot.key] = snapshot
        return snapshots

    def build_retrieval_cases(
        self,
        repo_name: str,
        pair: CommitPair,
        symbols: dict[tuple[str, str], SymbolSnapshot],
    ) -> list[RetrievalCase]:
        """Generate retrieval benchmark cases for indexed symbols."""
        cases: list[RetrievalCase] = []
        sorted_symbols = sorted(
            symbols.values(),
            key=lambda item: (item.relative_path, item.qualname),
        )
        for symbol in sorted_symbols:
            memory_id = self.memory_id(
                repo_name,
                pair.pair_id,
                symbol.relative_path,
                symbol.qualname,
            )
            cases.extend(
                (
                    RetrievalCase(
                        case_id=self._case_id(
                            "retrieval",
                            repo_name,
                            pair.pair_id,
                            symbol.relative_path,
                            symbol.qualname,
                            "taskish_behavior",
                        ),
                        repo_name=repo_name,
                        commit_pair_id=pair.pair_id,
                        slice_name="taskish_behavior",
                        memory_id=memory_id,
                        query=symbol.query,
                        expected_memory_ids=(memory_id,),
                        relative_path=symbol.relative_path,
                        symbol=symbol.qualname,
                    ),
                    RetrievalCase(
                        case_id=self._case_id(
                            "retrieval",
                            repo_name,
                            pair.pair_id,
                            symbol.relative_path,
                            symbol.qualname,
                            "smoke_identity",
                        ),
                        repo_name=repo_name,
                        commit_pair_id=pair.pair_id,
                        slice_name="smoke_identity",
                        memory_id=memory_id,
                        query=self._smoke_query_for_symbol(symbol),
                        expected_memory_ids=(memory_id,),
                        relative_path=symbol.relative_path,
                        symbol=symbol.qualname,
                    ),
                )
            )
        return cases

    def build_drift_cases(
        self,
        repo_name: str,
        pair: CommitPair,
        base_symbols: dict[tuple[str, str], SymbolSnapshot],
        head_symbols: dict[tuple[str, str], SymbolSnapshot],
    ) -> list[DriftCase]:
        """Generate drift cases by comparing base and head symbol snapshots."""
        cases: list[DriftCase] = []
        sorted_symbols = sorted(
            base_symbols.values(),
            key=lambda item: (item.relative_path, item.qualname),
        )
        for symbol in sorted_symbols:
            head_symbol = head_symbols.get(symbol.key)
            if head_symbol is None or head_symbol.symbol_digest != symbol.symbol_digest:
                expected_status = ValidityStatus.STALE
            elif (
                head_symbol.start_line == symbol.start_line
                and head_symbol.end_line == symbol.end_line
            ):
                expected_status = ValidityStatus.VERIFIED
            else:
                expected_status = ValidityStatus.RELOCATED

            cases.append(
                DriftCase(
                    case_id=self._case_id(
                        "drift",
                        repo_name,
                        pair.pair_id,
                        symbol.relative_path,
                        symbol.qualname,
                    ),
                    repo_name=repo_name,
                    commit_pair_id=pair.pair_id,
                    memory_id=self.memory_id(
                        repo_name,
                        pair.pair_id,
                        symbol.relative_path,
                        symbol.qualname,
                    ),
                    relative_path=symbol.relative_path,
                    symbol=symbol.qualname,
                    expected_status=expected_status,
                )
            )
        return cases

    def memory_id(
        self,
        repo_name: str,
        pair_id: str,
        relative_path: str,
        qualname: str,
    ) -> str:
        """Return the deterministic benchmark memory id for a symbol."""
        digest = hashlib.sha256(
            f"{repo_name}:{pair_id}:{relative_path}:{qualname}".encode()
        ).hexdigest()
        return f"mem_{digest[:24]}"

    def _extract_file_symbols(
        self,
        relative_path: str,
        source: str,
        nodes: list[ast.stmt],
        stack: tuple[str, ...] = (),
    ) -> list[SymbolSnapshot]:
        snapshots: list[SymbolSnapshot] = []
        for node in nodes:
            if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            qualname_parts = (*stack, node.name)
            qualname = ".".join(qualname_parts)
            if any(part.startswith("_") for part in qualname_parts):
                snapshots.extend(
                    self._extract_file_symbols(
                        relative_path,
                        source,
                        node.body,
                        qualname_parts,
                    )
                )
                continue
            docstring = ast.get_docstring(node)
            summary = self._summary_for_symbol(
                relative_path,
                qualname,
                self._kind_for_node(node),
                docstring,
            )
            query = self._query_for_symbol(relative_path, qualname, docstring)
            snippet = ast.get_source_segment(source, node) or self._slice_lines(
                source,
                node.lineno,
                getattr(node, "end_lineno", node.lineno) or node.lineno,
            )
            snapshots.append(
                SymbolSnapshot(
                    relative_path=relative_path,
                    qualname=qualname,
                    kind=self._kind_for_node(node),
                    start_line=node.lineno,
                    end_line=getattr(node, "end_lineno", node.lineno) or node.lineno,
                    symbol_digest=self._symbol_digest(node),
                    summary=summary,
                    query=query,
                    snippet=snippet,
                )
            )
            snapshots.extend(
                self._extract_file_symbols(
                    relative_path,
                    source,
                    node.body,
                    qualname_parts,
                )
            )
        return snapshots

    def _summary_for_symbol(
        self,
        relative_path: str,
        qualname: str,
        kind: str,
        docstring: str | None,
    ) -> str:
        sentence = self._first_sentence(docstring)
        if sentence is not None:
            return sentence
        return f"{kind.replace('_', ' ')} {qualname} defined in {relative_path}."

    def _query_for_symbol(self, relative_path: str, qualname: str, docstring: str | None) -> str:
        keywords = self._keywords_for_query(docstring)
        module_hint = Path(relative_path).stem.replace("_", " ")
        if keywords:
            return f"{self._task_prompt_for_path(relative_path, qualname)} {' '.join(keywords)}"
        return f"{self._task_prompt_for_path(relative_path, qualname)} {module_hint}".strip()

    def _smoke_query_for_symbol(self, symbol: SymbolSnapshot) -> str:
        parts = [
            symbol.qualname.replace(".", " "),
            Path(symbol.relative_path).stem.replace("_", " "),
        ]
        summary = self._first_sentence(symbol.summary)
        if summary is not None:
            parts.append(summary)
        return " ".join(parts)

    def _first_sentence(self, docstring: str | None) -> str | None:
        if docstring is None:
            return None
        normalized = " ".join(docstring.strip().split())
        if not normalized:
            return None
        return SENTENCE_RE.split(normalized, maxsplit=1)[0]

    def _slice_lines(self, source: str, start_line: int, end_line: int) -> str:
        lines = source.splitlines()
        return "\n".join(lines[start_line - 1 : end_line])

    def _kind_for_node(self, node: ast.AST) -> str:
        if isinstance(node, ast.ClassDef):
            return "class"
        if isinstance(node, ast.AsyncFunctionDef):
            return "async_function"
        return "function"

    def _keywords_for_query(self, docstring: str | None) -> tuple[str, ...]:
        sentence = self._first_sentence(docstring)
        if sentence is None:
            return ()
        keywords: list[str] = []
        for token in WORD_RE.findall(sentence.lower()):
            if token in STOPWORDS or token.isdigit():
                continue
            if token not in keywords:
                keywords.append(token)
        return tuple(keywords[:6])

    def _task_prompt_for_path(self, relative_path: str, qualname: str) -> str:
        kind = "class" if "." not in qualname and qualname[:1].isupper() else "function"
        module_hint = Path(relative_path).stem.replace("_", " ")
        return f"{kind} behavior in {module_hint}"

    def _case_id(
        self,
        case_type: str,
        repo_name: str,
        pair_id: str,
        relative_path: str,
        qualname: str,
        slice_name: str | None = None,
    ) -> str:
        parts = [case_type, repo_name, pair_id, relative_path, qualname]
        if slice_name is not None:
            parts.append(slice_name)
        return ":".join(parts)

    def _symbol_digest(self, node: ast.AST) -> str:
        payload = ast.dump(node, annotate_fields=True, include_attributes=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _is_test_path(self, relative_path: str) -> bool:
        parts = Path(relative_path).parts
        return "tests" in parts or Path(relative_path).name.startswith("test_")
