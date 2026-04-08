"""Runtime environment fingerprint collection helpers."""

from __future__ import annotations

import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from typing import Protocol

from vtm.fingerprints import EnvFingerprint, ToolVersion

DEFAULT_TOOL_PROBES: dict[str, tuple[str, ...]] = {
    "git": ("git", "--version"),
    "python3": ("python3", "--version"),
}


class EnvFingerprintAdapter(Protocol):
    """Interface for collecting environment fingerprints."""

    def collect(
        self,
        *,
        tool_probes: Mapping[str, Sequence[str]] | None = None,
    ) -> EnvFingerprint: ...


class RuntimeEnvFingerprintCollector:
    """Captures Python, platform, and probed tool versions."""

    def __init__(
        self,
        *,
        python_version: str | None = None,
        platform_name: str | None = None,
    ) -> None:
        """Create a collector with optional deterministic overrides."""
        self._python_version = python_version or platform.python_version()
        self._platform_name = platform_name or (
            f"{platform.system().lower()}-{platform.machine().lower()}"
        )

    def collect(
        self,
        *,
        tool_probes: Mapping[str, Sequence[str]] | None = None,
    ) -> EnvFingerprint:
        """Collect the runtime fingerprint using the configured probes."""
        probes = dict(DEFAULT_TOOL_PROBES)
        if tool_probes is not None:
            probes.update({name: tuple(command) for name, command in tool_probes.items()})
        tool_versions = tuple(
            ToolVersion(name=name, version=self._probe_version(name, command))
            for name, command in sorted(probes.items())
        )
        return EnvFingerprint(
            python_version=self._python_version,
            platform=self._platform_name,
            tool_versions=tool_versions,
        )

    def _probe_version(self, name: str, command: Sequence[str]) -> str:
        result = subprocess.run(
            list(command),
            check=True,
            capture_output=True,
            text=True,
        )
        first_line = (result.stdout or result.stderr).splitlines()[0].strip()
        return self._normalize_version(name, first_line)

    def _normalize_version(self, name: str, value: str) -> str:
        lowered_name = name.lower()
        lowered_value = value.lower()
        if lowered_value.startswith("git version "):
            return value[len("git version ") :]
        if lowered_value.startswith("python "):
            return value[len("Python ") :]
        if lowered_value.startswith(f"{lowered_name} "):
            return value[len(name) + 1 :]
        return value


def current_python_version() -> str:
    """Return the current interpreter version as `major.minor.patch`."""
    return ".".join(str(part) for part in sys.version_info[:3])
