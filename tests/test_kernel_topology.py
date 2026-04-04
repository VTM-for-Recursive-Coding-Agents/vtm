from __future__ import annotations

import pytest

from vtm.adapters.python_ast import PythonAstSyntaxAdapter
from vtm.services.memory_kernel import TransactionalMemoryKernel
from vtm.services.procedures import CommandProcedureValidator
from vtm.services.retriever import LexicalRetriever
from vtm.services.verifier import BasicVerifier


class InMemoryEventStore:
    def __init__(self) -> None:
        self._events = []

    def save_event(self, event) -> None:
        self._events.append(event)

    def get_event(self, event_id: str):
        for event in self._events:
            if event.event_id == event_id:
                return event
        return None

    def list_events(self):
        return tuple(self._events)


def test_kernel_requires_shared_sqlite_event_store_by_default(
    metadata_store,
    artifact_store,
    cache_store,
) -> None:
    with pytest.raises(ValueError, match="atomic event semantics require event_store"):
        TransactionalMemoryKernel(
            metadata_store=metadata_store,
            event_store=InMemoryEventStore(),
            artifact_store=artifact_store,
            cache_store=cache_store,
            verifier=BasicVerifier(relocator=PythonAstSyntaxAdapter()),
            retriever=LexicalRetriever(metadata_store),
            anchor_adapter=PythonAstSyntaxAdapter(),
            procedure_validator=CommandProcedureValidator(artifact_store),
        )


def test_kernel_allows_degraded_event_topology_when_opted_in(
    metadata_store,
    artifact_store,
    cache_store,
) -> None:
    kernel = TransactionalMemoryKernel(
        metadata_store=metadata_store,
        event_store=InMemoryEventStore(),
        artifact_store=artifact_store,
        cache_store=cache_store,
        verifier=BasicVerifier(relocator=PythonAstSyntaxAdapter()),
        retriever=LexicalRetriever(metadata_store),
        anchor_adapter=PythonAstSyntaxAdapter(),
        procedure_validator=CommandProcedureValidator(artifact_store),
        require_shared_event_store=False,
    )

    assert kernel is not None
