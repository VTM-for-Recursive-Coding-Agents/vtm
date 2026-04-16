"""Compatibility re-exports for the harness workspace surface."""

from vtm.harness.workspace import (
    CommandResult,
    LocalWorkspaceBackend,
    LocalWorkspaceDriver,
    PreparedWorkspace,
    WorkspaceBackend,
    WorkspaceDriver,
)
from vtm.harness.workspace_docker import DockerWorkspaceBackend, DockerWorkspaceDriver

__all__ = [
    "CommandResult",
    "DockerWorkspaceBackend",
    "DockerWorkspaceDriver",
    "LocalWorkspaceBackend",
    "LocalWorkspaceDriver",
    "PreparedWorkspace",
    "WorkspaceBackend",
    "WorkspaceDriver",
]
