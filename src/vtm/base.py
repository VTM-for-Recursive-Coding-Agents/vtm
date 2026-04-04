from __future__ import annotations

from datetime import UTC, datetime
from typing import TypeVar

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1.0"

ModelT = TypeVar("ModelT", bound="VTMModel")


def utc_now() -> datetime:
    return datetime.now(UTC)


class VTMModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(default=SCHEMA_VERSION)

    def to_json(self) -> str:
        return self.model_dump_json()

    @classmethod
    def from_json(cls: type[ModelT], payload: str) -> ModelT:
        return cls.model_validate_json(payload)
