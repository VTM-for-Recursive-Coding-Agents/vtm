# src/vtm/stores

Purpose: persistence layer for VTM. This package defines the storage protocols and the concrete SQLite/filesystem implementations used by the kernel.

Contents
- `__init__.py`: Re-exports the public store protocols and implementations.
- `_sqlite_schema.py`: Shared helpers for schema-version tracking and SQLite table inspection.
- `artifact_store.py`: Filesystem blob store with SQLite artifact metadata, capture lifecycle handling, and integrity audits.
- `base.py`: Store protocols for metadata, events, artifacts, cache, and embedding indexes.
- `cache_store.py`: SQLite cache implementation with expiry handling and optional cache hit/miss event logging.
- `embedding_store.py`: SQLite embedding index implementation keyed by memory id and embedding adapter id.
- `sqlite_store.py`: SQLite metadata and event store with transaction staging and JSONL export support.
- `migrations/`: Ordered schema migration entrypoints for metadata, cache, artifact, and embedding stores.
