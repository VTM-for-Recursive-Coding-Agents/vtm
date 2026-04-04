from __future__ import annotations

from datetime import datetime

from pydantic import Field

from vtm.base import VTMModel, utc_now


class EmbeddingIndexEntry(VTMModel):
    memory_id: str
    adapter_id: str
    content_digest: str
    vector: tuple[float, ...]
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
