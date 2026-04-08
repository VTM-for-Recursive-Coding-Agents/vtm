"""Shared enums that define durable kernel states and modes."""

from __future__ import annotations

from enum import StrEnum


class MemoryKind(StrEnum):
    """Top-level memory record categories."""

    CLAIM = "claim"
    PROCEDURE = "procedure"
    CONSTRAINT = "constraint"
    DECISION = "decision"
    SUMMARY_CARD = "summary_card"


class ValidityStatus(StrEnum):
    """Lifecycle states for verification and consolidation."""

    PENDING = "pending"
    VERIFIED = "verified"
    RELOCATED = "relocated"
    UNKNOWN = "unknown"
    STALE = "stale"
    REFUTED = "refuted"
    SUPERSEDED = "superseded"
    QUARANTINED = "quarantined"


class EvidenceKind(StrEnum):
    """Durable evidence reference targets."""

    ARTIFACT = "artifact"
    CODE_ANCHOR = "code_anchor"
    MEMORY = "memory"


class ArtifactCaptureState(StrEnum):
    """Lifecycle states for artifact capture records."""

    PREPARED = "prepared"
    COMMITTED = "committed"
    ABANDONED = "abandoned"


class ScopeKind(StrEnum):
    """Visibility namespaces supported by the kernel."""

    CALL = "call"
    TX = "tx"
    TASK = "task"
    BRANCH = "branch"
    REPO = "repo"
    USER = "user"
    GLOBAL = "global"


class ClaimStrength(StrEnum):
    """Strength hint attached to claim payloads."""

    TENTATIVE = "tentative"
    SUPPORTED = "supported"
    STRONG = "strong"
    CANONICAL = "canonical"


class TxState(StrEnum):
    """Transaction lifecycle states."""

    ACTIVE = "active"
    COMMITTED = "committed"
    ROLLED_BACK = "rolled_back"


class DetailLevel(StrEnum):
    """Preferred verbosity for retrieved memory payloads."""

    COMPACT = "compact"
    SUMMARY = "summary"
    FULL = "full"


class EvidenceBudget(StrEnum):
    """How much raw evidence retrieval should include."""

    SUMMARY_ONLY = "summary_only"
    SUMMARY_FIRST = "summary_first"
    FORCE_RAW = "force_raw"


class FreshnessMode(StrEnum):
    """How strict cache reuse should be."""

    STRICT = "strict"
    PREFER_FRESH = "prefer_fresh"
    ALLOW_STALE = "allow_stale"
