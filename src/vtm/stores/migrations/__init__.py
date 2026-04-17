"""Ordered SQLite schema migrations for VTM stores."""

from vtm.stores.migrations.artifact import (
    ARTIFACT_SCHEMA_VERSION,
    apply_artifact_migrations,
)
from vtm.stores.migrations.cache import CACHE_SCHEMA_VERSION, apply_cache_migrations
from vtm.stores.migrations.metadata import (
    METADATA_SCHEMA_VERSION,
    apply_metadata_migrations,
)

__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "CACHE_SCHEMA_VERSION",
    "METADATA_SCHEMA_VERSION",
    "apply_artifact_migrations",
    "apply_cache_migrations",
    "apply_metadata_migrations",
]
