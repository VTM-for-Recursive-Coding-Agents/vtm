"""Public benchmark manifest, result, and runner exports."""

from vtm.benchmarks.models import (
    BenchmarkCaseResult,
    BenchmarkComparisonResult,
    BenchmarkManifest,
    BenchmarkMatrixResult,
    BenchmarkRunConfig,
    BenchmarkRunResult,
    CodingTaskCase,
    CommitPair,
    DriftCase,
    RepoSpec,
    RetrievalCase,
)

__all__ = [
    "BenchmarkCaseResult",
    "BenchmarkComparisonResult",
    "BenchmarkManifest",
    "BenchmarkMatrixResult",
    "BenchmarkRunConfig",
    "BenchmarkRunResult",
    "BenchmarkRunner",
    "CodingTaskCase",
    "CommitPair",
    "DriftCase",
    "RepoSpec",
    "RetrievalCase",
]


def __getattr__(name: str) -> object:
    """Lazily import heavier benchmark helpers to avoid package cycles."""
    if name != "BenchmarkRunner":
        raise AttributeError(name)
    from vtm.benchmarks.runner import BenchmarkRunner

    return BenchmarkRunner
