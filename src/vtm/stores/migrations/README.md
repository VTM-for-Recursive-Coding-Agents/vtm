# src/vtm/stores/migrations

Purpose: ordered schema migration modules for each concrete store implementation.

Contents
- `__init__.py`: Re-exports the current schema versions and migration entrypoints.
- `artifact.py`: Artifact store migrations through schema version 2, including capture lifecycle columns and indexes.
- `cache.py`: Cache store schema version 1 definition.
- `metadata.py`: Metadata store migrations through schema version 2, including event export state tracking.
