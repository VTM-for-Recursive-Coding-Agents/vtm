"""Dependency fingerprint models used for verification and caching."""

from __future__ import annotations

from pydantic import Field

from vtm.base import VTMModel


class ToolVersion(VTMModel):
    """Version string captured for a runtime tool."""

    name: str
    version: str


class RepoFingerprint(VTMModel):
    """Snapshot of repository identity and tree state."""

    repo_root: str
    branch: str | None = None
    head_commit: str | None = None
    tree_digest: str | None = None
    dirty_digest: str | None = None


class EnvFingerprint(VTMModel):
    """Snapshot of relevant runtime environment details."""

    python_version: str
    platform: str
    tool_versions: tuple[ToolVersion, ...] = Field(default_factory=tuple)


class DependencyFingerprint(VTMModel):
    """Combined repository, environment, and input fingerprint."""

    repo: RepoFingerprint
    env: EnvFingerprint
    dependency_ids: tuple[str, ...] = Field(default_factory=tuple)
    input_digests: tuple[str, ...] = Field(default_factory=tuple)
