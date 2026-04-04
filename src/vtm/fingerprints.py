from __future__ import annotations

from pydantic import Field

from vtm.base import VTMModel


class ToolVersion(VTMModel):
    name: str
    version: str


class RepoFingerprint(VTMModel):
    repo_root: str
    branch: str | None = None
    head_commit: str | None = None
    tree_digest: str | None = None
    dirty_digest: str | None = None


class EnvFingerprint(VTMModel):
    python_version: str
    platform: str
    tool_versions: tuple[ToolVersion, ...] = Field(default_factory=tuple)


class DependencyFingerprint(VTMModel):
    repo: RepoFingerprint
    env: EnvFingerprint
    dependency_ids: tuple[str, ...] = Field(default_factory=tuple)
    input_digests: tuple[str, ...] = Field(default_factory=tuple)
