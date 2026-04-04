from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime
from hashlib import sha256
from typing import Any

from pydantic import BaseModel, Field, model_validator

from vtm.base import VTMModel, utc_now
from vtm.enums import FreshnessMode
from vtm.fingerprints import EnvFingerprint, RepoFingerprint
from vtm.ids import new_cache_entry_id


def _normalize_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {key: _normalize_value(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_normalize_value(item) for item in value]
    return value


def _normalize_args(args: Mapping[str, Any]) -> str:
    normalized = {key: _normalize_value(args[key]) for key in sorted(args)}
    return json.dumps(normalized, separators=(",", ":"), sort_keys=True)


class CacheKey(VTMModel):
    tool_name: str
    normalized_args_json: str
    repo_fingerprint: RepoFingerprint
    env_fingerprint: EnvFingerprint
    digest: str

    @classmethod
    def from_parts(
        cls,
        tool_name: str,
        args: Mapping[str, Any],
        repo_fingerprint: RepoFingerprint,
        env_fingerprint: EnvFingerprint,
    ) -> CacheKey:
        normalized_args_json = _normalize_args(args)
        digest_payload = json.dumps(
            {
                "tool_name": tool_name,
                "args": json.loads(normalized_args_json),
                "repo_fingerprint": repo_fingerprint.model_dump(mode="json"),
                "env_fingerprint": env_fingerprint.model_dump(mode="json"),
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        return cls(
            tool_name=tool_name,
            normalized_args_json=normalized_args_json,
            repo_fingerprint=repo_fingerprint,
            env_fingerprint=env_fingerprint,
            digest=sha256(digest_payload.encode("utf-8")).hexdigest(),
        )


class CacheEntry(VTMModel):
    entry_id: str = Field(default_factory=new_cache_entry_id)
    key: CacheKey
    value: dict[str, Any] = Field(default_factory=dict)
    freshness_mode: FreshnessMode = FreshnessMode.PREFER_FRESH
    created_at: datetime = Field(default_factory=utc_now)
    expires_at: datetime | None = None
    hit_count: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_expiry(self) -> CacheEntry:
        if self.expires_at is not None and self.expires_at < self.created_at:
            raise ValueError("cache entries cannot expire before they are created")
        return self
