"""Public harness contracts and local reference implementations."""

from vtm.harness.models import (
    ExecutorRequest,
    ExecutorResult,
    HarnessTaskPack,
    TaskMemoryContextItem,
    TraceManifest,
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

__all__ = [
    "BenchmarkExecutor",
    "CommandResult",
    "ExecutorRequest",
    "ExecutorResult",
    "HarnessTaskPack",
    "LocalWorkspaceBackend",
    "LocalWorkspaceDriver",
    "NativeAgentBenchmarkExecutor",
    "PreparedWorkspace",
    "SubprocessBenchmarkExecutor",
    "TaskMemoryContextItem",
    "TraceManifest",
    "WorkspaceBackend",
    "WorkspaceDriver",
    "changed_path_metrics",
    "patch_similarity",
]


def __getattr__(name: str) -> object:
    """Lazily import executor implementations to keep base imports light."""
    if name not in {
        "BenchmarkExecutor",
        "NativeAgentBenchmarkExecutor",
        "SubprocessBenchmarkExecutor",
    }:
        raise AttributeError(name)
    from vtm.harness.executors import (
        BenchmarkExecutor,
        NativeAgentBenchmarkExecutor,
        SubprocessBenchmarkExecutor,
    )

    mapping = {
        "BenchmarkExecutor": BenchmarkExecutor,
        "NativeAgentBenchmarkExecutor": NativeAgentBenchmarkExecutor,
        "SubprocessBenchmarkExecutor": SubprocessBenchmarkExecutor,
    }
    return mapping[name]
