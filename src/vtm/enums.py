from __future__ import annotations

from enum import StrEnum


class MemoryKind(StrEnum):
    CLAIM = "claim"
    PROCEDURE = "procedure"
    CONSTRAINT = "constraint"
    DECISION = "decision"
    SUMMARY_CARD = "summary_card"


class ValidityStatus(StrEnum):
    PENDING = "pending"
    VERIFIED = "verified"
    RELOCATED = "relocated"
    UNKNOWN = "unknown"
    STALE = "stale"
    REFUTED = "refuted"
    SUPERSEDED = "superseded"
    QUARANTINED = "quarantined"


class EvidenceKind(StrEnum):
    ARTIFACT = "artifact"
    CODE_ANCHOR = "code_anchor"
    MEMORY = "memory"


class ArtifactCaptureState(StrEnum):
    PREPARED = "prepared"
    COMMITTED = "committed"
    ABANDONED = "abandoned"


class ScopeKind(StrEnum):
    CALL = "call"
    TX = "tx"
    TASK = "task"
    BRANCH = "branch"
    REPO = "repo"
    USER = "user"
    GLOBAL = "global"


class ClaimStrength(StrEnum):
    TENTATIVE = "tentative"
    SUPPORTED = "supported"
    STRONG = "strong"
    CANONICAL = "canonical"


class TxState(StrEnum):
    ACTIVE = "active"
    COMMITTED = "committed"
    ROLLED_BACK = "rolled_back"


class DetailLevel(StrEnum):
    COMPACT = "compact"
    SUMMARY = "summary"
    FULL = "full"


class EvidenceBudget(StrEnum):
    SUMMARY_ONLY = "summary_only"
    SUMMARY_FIRST = "summary_first"
    FORCE_RAW = "force_raw"


class FreshnessMode(StrEnum):
    STRICT = "strict"
    PREFER_FRESH = "prefer_fresh"
    ALLOW_STALE = "allow_stale"
