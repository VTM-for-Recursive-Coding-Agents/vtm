"""Canonical event-ledger records emitted by kernel operations."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import Field

from vtm.base import VTMModel, utc_now
from vtm.ids import new_event_id


class MemoryEvent(VTMModel):
    """An append-only event describing a kernel-side mutation."""

    event_id: str = Field(default_factory=new_event_id)
    event_type: str
    occurred_at: datetime = Field(default_factory=utc_now)
    tx_id: str | None = None
    memory_id: str | None = None
    cache_digest: str | None = None
    actor: str = "system"
    session_id: str | None = None
    tool_name: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
