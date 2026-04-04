from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar, cast

from vtm.events import MemoryEvent
from vtm.stores.base import EventStore, MetadataStore

MutationResultT = TypeVar("MutationResultT")


class MetadataMutationRunner:
    def __init__(self, *, metadata_store: MetadataStore, event_store: EventStore) -> None:
        self._metadata_store = metadata_store
        self._event_store = event_store

    def run(
        self,
        operation: Callable[[], MutationResultT],
        *,
        build_events: Callable[[MutationResultT], tuple[MemoryEvent, ...]] | None = None,
    ) -> MutationResultT:
        if cast(object, self._event_store) is cast(object, self._metadata_store):

            def wrapped() -> MutationResultT:
                result = operation()
                if build_events is not None:
                    for event in build_events(result):
                        self._event_store.save_event(event)
                return result

            return self._metadata_store.run_atomically(wrapped)

        result = self._metadata_store.run_atomically(operation)
        if build_events is not None:
            for event in build_events(result):
                self._event_store.save_event(event)
        return result
