from __future__ import annotations

from typing import NewType
from uuid import uuid4

MemoryId = NewType("MemoryId", str)
ArtifactId = NewType("ArtifactId", str)
TransactionId = NewType("TransactionId", str)
EventId = NewType("EventId", str)
CacheEntryId = NewType("CacheEntryId", str)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


def new_memory_id() -> MemoryId:
    return MemoryId(_new_id("mem"))


def new_artifact_id() -> ArtifactId:
    return ArtifactId(_new_id("art"))


def new_transaction_id() -> TransactionId:
    return TransactionId(_new_id("tx"))


def new_event_id() -> EventId:
    return EventId(_new_id("evt"))


def new_cache_entry_id() -> CacheEntryId:
    return CacheEntryId(_new_id("cache"))
