"""Runtime context shared by the benchmark harness and vendored RLM integration."""

from __future__ import annotations

from dataclasses import dataclass

from vtm.memory_items import VisibilityScope
from vtm.services import DependencyFingerprintBuilder, TransactionalMemoryKernel


@dataclass(frozen=True)
class RLMRuntimeContext:
    """Execution-time dependencies needed by the vendored RLM bridge."""

    kernel: TransactionalMemoryKernel | None
    task_scope: VisibilityScope | None
    durable_scope: VisibilityScope | None
    dependency_builder: DependencyFingerprintBuilder | None


__all__ = ["RLMRuntimeContext"]
