"""Code-anchor records and protocols for symbol-oriented evidence."""

from __future__ import annotations

from typing import Protocol

from pydantic import AliasChoices, Field

from vtm.base import VTMModel


class CodeAnchor(VTMModel):
    """Locates a symbol or span inside a source file."""

    path: str
    symbol: str | None = None
    kind: str | None = None
    language: str | None = None
    symbol_digest: str | None = Field(
        default=None,
        validation_alias=AliasChoices("symbol_digest", "ast_digest"),
        serialization_alias="symbol_digest",
    )
    context_digest: str | None = None
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    start_byte: int | None = Field(default=None, ge=0)
    end_byte: int | None = Field(default=None, ge=0)

    @property
    def ast_digest(self) -> str | None:
        """Compatibility alias for older `ast_digest` field names."""
        return self.symbol_digest


class AnchorRelocation(VTMModel):
    """Describes how an existing anchor was relocated after source changes."""

    old_anchor: CodeAnchor
    new_anchor: CodeAnchor
    method: str
    confidence: float = Field(ge=0.0, le=1.0)


class AnchorVerifier(Protocol):
    """Checks whether a stored anchor is still valid in the current source tree."""

    def verify(self, anchor: CodeAnchor) -> bool: ...


class AnchorRelocator(Protocol):
    """Attempts to move a stale anchor onto the current source span."""

    def relocate(self, anchor: CodeAnchor) -> AnchorRelocation | None: ...


class AnchorAdapter(Protocol):
    """Combined build-and-relocate contract used by the kernel."""

    def build_anchor(self, source_path: str, symbol: str) -> CodeAnchor: ...

    def relocate(self, anchor: CodeAnchor) -> AnchorRelocation | None: ...
