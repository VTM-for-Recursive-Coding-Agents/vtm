"""Public harness contracts and local reference implementations."""

from vtm.harness.models import (
    ExecutorRequest,
    ExecutorResult,
    HarnessTaskPack,
    TaskMemoryContextItem,
)
from vtm.harness.scoring import changed_path_metrics, patch_similarity
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
    "BenchmarkExecutor",
    "CommandResult",
    "DockerWorkspaceBackend",
    "DockerWorkspaceDriver",
    "ExecutorRequest",
    "ExecutorResult",
    "HarnessTaskPack",
    "LocalWorkspaceBackend",
    "LocalWorkspaceDriver",
    "PreparedWorkspace",
    "RLMBenchmarkExecutor",
    "TaskMemoryContextItem",
    "WorkspaceBackend",
    "WorkspaceDriver",
    "changed_path_metrics",
    "patch_similarity",
]


def __getattr__(name: str) -> object:
    """Lazily import executor implementations to keep base imports light."""
    if name not in {
        "BenchmarkExecutor",
        "RLMBenchmarkExecutor",
    }:
        raise AttributeError(name)
    from vtm.harness.executors import (
        BenchmarkExecutor,
        RLMBenchmarkExecutor,
    )

    mapping = {
        "BenchmarkExecutor": BenchmarkExecutor,
        "RLMBenchmarkExecutor": RLMBenchmarkExecutor,
    }
    return mapping[name]
