"""Default retrieval policy constants."""

from __future__ import annotations

from vtm.enums import ValidityStatus

DEFAULT_RETRIEVAL_STATUSES: tuple[ValidityStatus, ...] = (
    ValidityStatus.VERIFIED,
    ValidityStatus.RELOCATED,
)
