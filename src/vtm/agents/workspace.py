"""Compatibility re-exports for the harness workspace surface."""

from vtm.harness.workspace import (
    CommandResult,
    LocalWorkspaceBackend,
    LocalWorkspaceDriver,
    PreparedWorkspace,
    WorkspaceBackend,
    WorkspaceDriver,
)

__all__ = [
    "CommandResult",
    "LocalWorkspaceBackend",
    "LocalWorkspaceDriver",
    "PreparedWorkspace",
    "WorkspaceBackend",
    "WorkspaceDriver",
]
