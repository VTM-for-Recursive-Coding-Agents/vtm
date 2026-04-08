"""Dependency fingerprint construction helpers."""

from __future__ import annotations

from collections.abc import Sequence

from vtm.adapters.git import GitFingerprintAdapter
from vtm.adapters.runtime import EnvFingerprintAdapter
from vtm.fingerprints import DependencyFingerprint


class DependencyFingerprintBuilder:
    """Builds combined repo and environment fingerprints for the kernel."""

    def __init__(
        self,
        *,
        repo_collector: GitFingerprintAdapter,
        env_collector: EnvFingerprintAdapter,
    ) -> None:
        """Create a builder from repo and environment collectors."""
        self._repo_collector = repo_collector
        self._env_collector = env_collector

    def build(
        self,
        repo_root: str,
        *,
        dependency_ids: Sequence[str] = (),
        input_digests: Sequence[str] = (),
        tool_probes: dict[str, tuple[str, ...]] | None = None,
    ) -> DependencyFingerprint:
        """Collect and combine repo, env, and caller-supplied dependency inputs."""
        return DependencyFingerprint(
            repo=self._repo_collector.collect(repo_root),
            env=self._env_collector.collect(tool_probes=tool_probes),
            dependency_ids=tuple(dependency_ids),
            input_digests=tuple(input_digests),
        )
