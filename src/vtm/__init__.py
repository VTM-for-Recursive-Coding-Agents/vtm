"""Kernel-first public export surface for VTM.

Import stable memory, storage, retrieval, verification, and consolidation
types from here. Harness, agent-runtime, and benchmark-specific surfaces live
under their owning subpackages.
"""

from __future__ import annotations

import importlib

from vtm.adapters.embeddings import DeterministicHashEmbeddingAdapter, EmbeddingAdapter
from vtm.adapters.git import GitFingerprintAdapter, GitRepoFingerprintCollector
from vtm.adapters.python_ast import (
    PythonAstAnchorAdapter,
    PythonAstAnchorRelocator,
    PythonAstSyntaxAdapter,
)
from vtm.adapters.rlm import RLMAdapter, RLMRankedCandidate, RLMRankRequest, RLMRankResponse
from vtm.adapters.runtime import (
    DEFAULT_TOOL_PROBES,
    EnvFingerprintAdapter,
    RuntimeEnvFingerprintCollector,
)
from vtm.adapters.tree_sitter import (
    PythonTreeSitterSyntaxAdapter,
    SyntaxAnchorAdapter,
    SyntaxTreeAdapter,
    UnavailableTreeSitterAdapter,
)
from vtm.anchors import AnchorAdapter, AnchorRelocation, AnchorRelocator, AnchorVerifier, CodeAnchor
from vtm.artifacts import ArtifactIntegrityReport, ArtifactRecord
from vtm.base import SCHEMA_VERSION, VTMModel
from vtm.cache import CacheEntry, CacheKey
from vtm.consolidation import ConsolidationAction, ConsolidationRunResult
from vtm.embeddings import EmbeddingIndexEntry
from vtm.enums import (
    ArtifactCaptureState,
    ClaimStrength,
    DetailLevel,
    EvidenceBudget,
    EvidenceKind,
    FreshnessMode,
    MemoryKind,
    ScopeKind,
    TxState,
    ValidityStatus,
)
from vtm.events import MemoryEvent
from vtm.evidence import ArtifactRef, EvidenceRef
from vtm.fingerprints import DependencyFingerprint, EnvFingerprint, RepoFingerprint, ToolVersion
from vtm.memory_items import (
    ClaimPayload,
    ConstraintPayload,
    DecisionPayload,
    LineageEdge,
    MemoryItem,
    MemoryStats,
    ProcedurePayload,
    ProcedureStep,
    SummaryCardPayload,
    ValidatorSpec,
    ValidityState,
    VisibilityScope,
)
from vtm.retrieval import RetrieveCandidate, RetrieveExplanation, RetrieveRequest, RetrieveResult
from vtm.services import (
    BasicVerifier,
    CommandProcedureValidator,
    Consolidator,
    DependencyFingerprintBuilder,
    DeterministicConsolidator,
    EmbeddingRetriever,
    LexicalRetriever,
    MemoryKernel,
    NoopConsolidator,
    ProcedureValidator,
    Retriever,
    RLMRerankingRetriever,
    TransactionalMemoryKernel,
    Verifier,
)
from vtm.stores import (
    ArtifactStore,
    CacheStore,
    EmbeddingIndexStore,
    EventStore,
    FilesystemArtifactStore,
    MetadataStore,
    SqliteCacheStore,
    SqliteEmbeddingIndexStore,
    SqliteMetadataStore,
)
from vtm.transactions import TransactionRecord
from vtm.verification import ProcedureValidationResult, VerificationResult

__all__ = [
    "DEFAULT_TOOL_PROBES",
    "AnchorAdapter",
    "AnchorRelocation",
    "AnchorRelocator",
    "AnchorVerifier",
    "ArtifactCaptureState",
    "ArtifactIntegrityReport",
    "ArtifactRecord",
    "ArtifactRef",
    "ArtifactStore",
    "BasicVerifier",
    "CacheEntry",
    "CacheKey",
    "CacheStore",
    "ClaimPayload",
    "ClaimStrength",
    "CodeAnchor",
    "CommandProcedureValidator",
    "ConsolidationAction",
    "ConsolidationRunResult",
    "Consolidator",
    "ConstraintPayload",
    "DecisionPayload",
    "DependencyFingerprint",
    "DependencyFingerprintBuilder",
    "DetailLevel",
    "DeterministicConsolidator",
    "DeterministicHashEmbeddingAdapter",
    "EmbeddingAdapter",
    "EmbeddingIndexEntry",
    "EmbeddingIndexStore",
    "EmbeddingRetriever",
    "EnvFingerprint",
    "EnvFingerprintAdapter",
    "EvidenceBudget",
    "EvidenceKind",
    "EvidenceRef",
    "EventStore",
    "FilesystemArtifactStore",
    "FreshnessMode",
    "GitFingerprintAdapter",
    "GitRepoFingerprintCollector",
    "LexicalRetriever",
    "LineageEdge",
    "MemoryEvent",
    "MemoryItem",
    "MemoryKernel",
    "MemoryKind",
    "MemoryStats",
    "MetadataStore",
    "NoopConsolidator",
    "ProcedurePayload",
    "ProcedureStep",
    "ProcedureValidationResult",
    "ProcedureValidator",
    "PythonAstAnchorAdapter",
    "PythonAstAnchorRelocator",
    "PythonAstSyntaxAdapter",
    "PythonTreeSitterSyntaxAdapter",
    "RLMAdapter",
    "RLMRankRequest",
    "RLMRankResponse",
    "RLMRankedCandidate",
    "RLMRerankingRetriever",
    "RepoFingerprint",
    "RetrieveCandidate",
    "RetrieveExplanation",
    "RetrieveRequest",
    "RetrieveResult",
    "Retriever",
    "RuntimeEnvFingerprintCollector",
    "SCHEMA_VERSION",
    "ScopeKind",
    "SqliteCacheStore",
    "SqliteEmbeddingIndexStore",
    "SqliteMetadataStore",
    "SummaryCardPayload",
    "SyntaxAnchorAdapter",
    "SyntaxTreeAdapter",
    "ToolVersion",
    "TransactionRecord",
    "TransactionalMemoryKernel",
    "TxState",
    "UnavailableTreeSitterAdapter",
    "ValidatorSpec",
    "ValidityState",
    "ValidityStatus",
    "VerificationResult",
    "Verifier",
    "VisibilityScope",
    "VTMModel",
]


def __getattr__(name: str) -> object:
    """Lazily resolve compatibility exports for optional or moved surfaces."""
    benchmark_names = {
        "BenchmarkCaseResult",
        "BenchmarkManifest",
        "BenchmarkRunConfig",
        "BenchmarkRunResult",
        "BenchmarkRunner",
        "CodingTaskCase",
        "CommitPair",
        "DriftCase",
        "RepoSpec",
        "RetrievalCase",
    }
    if name in benchmark_names:
        benchmark_module = importlib.import_module("vtm.benchmarks")
        return getattr(benchmark_module, name)

    if name in {"OpenAIEmbeddingAdapter", "OpenAIRLMAdapter"}:
        adapter_module = importlib.import_module("vtm.adapters")
        return getattr(adapter_module, name)

    raise AttributeError(name)
