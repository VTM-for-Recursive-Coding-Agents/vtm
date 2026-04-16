# API Notes

This document covers the stable kernel-facing API.

Benchmark orchestration lives under `vtm.benchmarks`. Executor and task-pack contracts live under `vtm.harness`. Vendored-RLM integration lives under `vtm_rlm`.

## Recommended root imports

```python
from vtm import (
    FilesystemArtifactStore,
    LexicalRetriever,
    SqliteMetadataStore,
    TransactionalMemoryKernel,
)
```

The root `vtm` package is intentionally kernel-first. Benchmark and provider-specific helpers may still be reachable for compatibility, but new code should import them from their owning subpackages.

## Typical kernel wiring

The smallest practical kernel topology uses a shared SQLite metadata/event store, a filesystem artifact store, a SQLite cache store, a verifier, and a retriever:

```python
from pathlib import Path

from vtm import LexicalRetriever, TransactionalMemoryKernel
from vtm.adapters import PythonAstSyntaxAdapter, PythonTreeSitterSyntaxAdapter
from vtm.services import BasicVerifier
from vtm.stores import FilesystemArtifactStore, SqliteCacheStore, SqliteMetadataStore

repo_root = Path(".").resolve()
metadata = SqliteMetadataStore(
    repo_root / ".vtm" / "metadata.sqlite",
    event_log_path=repo_root / ".vtm" / "events.jsonl",
)
artifacts = FilesystemArtifactStore(repo_root / ".vtm" / "artifacts")
cache = SqliteCacheStore(repo_root / ".vtm" / "cache.sqlite", event_store=metadata)
anchor_adapter = PythonTreeSitterSyntaxAdapter(fallback=PythonAstSyntaxAdapter())

kernel = TransactionalMemoryKernel(
    metadata_store=metadata,
    event_store=metadata,
    artifact_store=artifacts,
    cache_store=cache,
    verifier=BasicVerifier(relocator=anchor_adapter),
    retriever=LexicalRetriever(metadata),
    anchor_adapter=anchor_adapter,
)
```

That topology preserves atomic SQLite semantics for metadata and events because `event_store is metadata_store`.

## Choosing imports

- Import from `vtm` for stable kernel records, stores, and services.
- Import from `vtm.harness` for task packs, workspaces, executors, and scoring contracts.
- Import from `vtm_rlm` for vendored-RLM execution helpers and runtime context.
- Import from `vtm.benchmarks` for manifest models and `BenchmarkRunner`.
- Import from `vtm.adapters` only when you need provider or environment integrations directly.

## Store protocols

- `MetadataStore`
  - save/get/list/query memory items
  - save/list lineage edges
  - save/get/list transactions
  - append/list/move/clear staged memory items
  - run grouped metadata mutations through `run_atomically(...)`
- `EventStore`
  - save/get/list memory events
- `ArtifactStore`
  - prepare, commit, abandon, and read artifact captures
  - fetch the latest artifact record by digest
  - audit prepared captures, missing committed blobs, and orphaned blobs
- `CacheStore`
  - save/get/delete/list cache entries
- `EmbeddingIndexStore`
  - save/get/list/delete derived embedding rows keyed by `(memory_id, adapter_id)`

## Kernel services

- `TransactionalMemoryKernel`
  - `begin_transaction`
  - `stage_memory_item`
  - `list_visible_memory`
  - `commit_transaction`
  - `rollback_transaction`
  - `build_code_anchor`
  - `capture_artifact`
  - `artifact_evidence`
  - `anchor_evidence`
  - `verify_memory`
  - `validate_procedure`
  - `promote_to_procedure`
  - `retrieve`
  - `expand`
  - `save_cache_entry` / `get_cache_entry`
- `DependencyFingerprintBuilder`
  - combines repo and environment collectors with caller-supplied dependency IDs and input digests
- `BasicVerifier`
  - verifies dependency fingerprints and optionally relocates code anchors
- `LexicalRetriever`
  - deterministic lexical ranking over committed memory
- `EmbeddingRetriever`
  - derived-vector retrieval over the same committed-memory surface
- `RLMRerankingRetriever`
  - wrapper over an existing retriever using `RLMAdapter`
- `CommandProcedureValidator`
  - command-based validation with committed stdout/stderr artifact capture
  - consumes typed `CommandValidatorConfig` models through `ValidatorSpec(kind="command")`
  - supports optional cwd confinement, parent-env suppression, and POSIX resource limits for tighter local execution
- `DockerProcedureValidator`
  - command-based validation through `DockerWorkspaceBackend`
  - snapshots the current repo working tree into a Docker-backed workspace and records container metadata on the validation result
  - confines execution to `repo_root` and reuses the harness Docker sandbox defaults
- `DeterministicConsolidator`
  - deterministic duplicate superseding and optional summary-card generation

`TransactionalMemoryKernel(...)` defaults to `require_shared_event_store=True`. In the default topology, `event_store is metadata_store` and both are `SqliteMetadataStore`, so transaction state changes, verification writeback, procedure validation writeback, retrieval-stat updates, and corresponding event rows are atomic in SQLite.

## Concrete kernel implementations

- `SqliteMetadataStore`
- `FilesystemArtifactStore`
- `SqliteCacheStore`
- `SqliteEmbeddingIndexStore`
- `GitRepoFingerprintCollector`
- `RuntimeEnvFingerprintCollector`
- `PythonTreeSitterSyntaxAdapter(fallback=PythonAstSyntaxAdapter())`
- `PythonAstSyntaxAdapter()`
- `DeterministicHashEmbeddingAdapter(dimensions=64)`

Optional provider-specific implementations live in `vtm.adapters`:

- `OpenAIEmbeddingAdapter`
- `OpenAIRLMAdapter`

## Recovery model

- SQLite is the canonical event ledger.
- JSONL event export is still derived and repaired through `rebuild_events_jsonl()`, but export now resumes from complete on-disk lines before appending new rows.
- Artifact capture uses explicit prepared/committed/abandoned states so recovery is inspectable through `audit_integrity()`.
- Abandoned captures now retain structured provenance in `metadata["abandon_provenance"]`, so operator tooling can distinguish kernel capture writeback fallout, procedure-validation writeback fallout, validator-side capture fallout, and janitor cleanup.
- `repair_integrity()` applies the safe repair steps in one pass: abandon lingering prepared captures and delete orphaned blobs, while leaving committed-missing-blob cases as unresolved diagnostics.

## Related packages

- `vtm.harness`: typed task packs, workspace backends, executors, and scoring helpers
- `vtm_rlm`: vendored-RLM bridge, runtime context, prompt shaping, and writeback helpers
- `vtm.benchmarks`: manifests, runner, reporting, and SWE-bench preparation/integration
