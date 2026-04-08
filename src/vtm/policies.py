"""Default policy constants shared across retrieval and caching."""

from __future__ import annotations

from vtm.enums import FreshnessMode, ScopeKind, ValidityStatus

DEFAULT_RETRIEVAL_STATUSES: tuple[ValidityStatus, ...] = (
    ValidityStatus.VERIFIED,
    ValidityStatus.RELOCATED,
)
DEFAULT_FRESHNESS_MODE = FreshnessMode.PREFER_FRESH

SCOPE_ORDER: tuple[ScopeKind, ...] = (
    ScopeKind.CALL,
    ScopeKind.TX,
    ScopeKind.TASK,
    ScopeKind.BRANCH,
    ScopeKind.REPO,
    ScopeKind.USER,
    ScopeKind.GLOBAL,
)


def is_default_retrievable(status: ValidityStatus) -> bool:
    """Return whether a validity status is retrievable by default."""
    return status in DEFAULT_RETRIEVAL_STATUSES
