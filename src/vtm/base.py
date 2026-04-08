"""Shared model and timestamp helpers used across VTM records."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TypeVar

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1.0"

ModelT = TypeVar("ModelT", bound="VTMModel")


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(UTC)


class VTMModel(BaseModel):
    """Common immutable Pydantic base for durable VTM records."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(default=SCHEMA_VERSION)

    def to_json(self) -> str:
        """Serialize the model using the durable JSON representation."""
        return self.model_dump_json()

    @classmethod
    def from_json(cls: type[ModelT], payload: str) -> ModelT:
        """Deserialize a durable JSON payload into the concrete model type."""
        return cls.model_validate_json(payload)
