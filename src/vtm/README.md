# src/vtm

Purpose: public VTM package. This directory holds the stable record layer plus the service, store, adapter, and benchmark subpackages.

Contents
- `__init__.py`: Package export surface that re-exports the main public types and implementations.
- `anchors.py`: Code-anchor models plus protocols for anchor verification and relocation.
- `artifacts.py`: Artifact capture record and artifact integrity report models.
- `base.py`: Shared base model, schema version constant, and common UTC timestamp helper.
- `cache.py`: Cache key normalization plus cache entry models used by cache-backed services.
- `consolidation.py`: Consolidation action and run-result records for deterministic maintenance passes.
- `embeddings.py`: Derived embedding index record used by embedding-backed retrieval.
- `enums.py`: Shared string enums for memory kinds, statuses, scopes, capture states, and retrieval policy.
- `events.py`: Canonical memory-event record used by metadata and cache event logging.
- `evidence.py`: Typed evidence references for artifacts, code anchors, and other memories.
- `fingerprints.py`: Repo, environment, tool-version, and dependency fingerprint record types.
- `ids.py`: Typed ID aliases and UUID-based ID constructors for persisted records.
- `memory_items.py`: Core memory payloads, visibility, validity, lineage, stats, and `MemoryItem` invariants.
- `policies.py`: Default retrieval policy constants and helper functions.
- `retrieval.py`: Retrieval request, explanation, candidate, and result record types.
- `transactions.py`: Transaction record model and transaction-state validation.
- `verification.py`: Verification and procedure-validation result records.
- `adapters/`: Integration boundary for Git, runtime, parser, embedding, and reranking adapters.
- `benchmarks/`: Benchmark harness package for retrieval, drift, and coding-task evaluation.
- `services/`: Kernel orchestration layer and retriever, validator, and consolidator implementations.
- `stores/`: Persistence layer protocols and concrete SQLite/filesystem stores.
