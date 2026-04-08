# src/vtm/stores

Purpose: persistence layer for the kernel.

Contents
- `base.py`: Store protocols.
- `sqlite_store.py`: Metadata and canonical event store.
- `cache_store.py`: Deterministic cache store.
- `embedding_store.py`: Derived embedding index store.
- `artifact_store.py`: Filesystem blob store plus capture metadata.
- `_sqlite_schema.py` and `migrations/`: Schema tracking and ordered migrations.
