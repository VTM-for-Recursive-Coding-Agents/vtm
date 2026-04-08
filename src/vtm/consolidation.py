"""Result records produced by consolidation runs."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import Field

from vtm.base import VTMModel, utc_now

ConsolidationActionType = Literal["memory_superseded", "summary_card_created"]


class ConsolidationAction(VTMModel):
    """Single durable change emitted by a consolidator."""

    action_type: ConsolidationActionType
    canonical_memory_id: str
    affected_memory_ids: tuple[str, ...] = Field(default_factory=tuple)
    created_memory_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConsolidationRunResult(VTMModel):
    """Aggregate outcome for one consolidation pass."""

    scanned_memory_count: int = Field(ge=0)
    candidate_group_count: int = Field(ge=0)
    action_count: int = Field(ge=0)
    actions: tuple[ConsolidationAction, ...] = Field(default_factory=tuple)
    started_at: datetime = Field(default_factory=utc_now)
    completed_at: datetime = Field(default_factory=utc_now)
