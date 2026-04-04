# Current State Audit

## Purpose and scope

This document is the maintainer-facing status snapshot for the current VTM kernel. It answers three questions:

- What behavior is implemented and verified today?
- Which correctness and auditability gaps still remain?
- What work is intentionally limited rather than missing?

The focus here is the concrete local deployment that now ships SQLite-backed metadata, cache, and embedding stores, a filesystem artifact store, deterministic lexical and embedding retrieval, optional RLM reranking, deterministic consolidation, local command-based procedure validation, and a benchmark harness for retrieval, drift, synthetic coding tasks, and SWE-bench Lite-backed coding evaluation.

## Verified baseline as of April 3, 2026

- `uv run pytest -q` passes with 118 tests.
- `uv run python -m mypy src` passes in strict mode.
- `uv run python -m ruff check .` passes.
- Benchmark coverage exercises lexical retrieval, embedding retrieval, drift, reranking through the benchmark runner, coding-task dry runs, executor artifact capture, duplicate-symbol case-ID regression checks, repo/pair filtering, SWE-bench Lite manifest preparation, local patcher behavior, and harness-backed coding aggregation.
- Synthetic coding coverage now exercises multiple deterministic local bug-fix tasks, task-pack metadata, changed-path scoring, mode comparisons, coding-task filter validation, and an end-to-end fake harness-backed SWE-bench path.
- Active transactions persist both transaction records and staged memory items in SQLite and survive kernel restarts.
- Root commit, child commit, rollback, verification, procedure validation metadata updates, procedure promotion, and retrieval-stat updates are atomic when the metadata store and event store are the same `SqliteMetadataStore` instance.
- SQLite stores track schema version state, reject unknown future schema versions, and have fixture-backed coverage for every supported metadata, cache, artifact, and embedding revision.
- The kernel rejects non-shared event-store topologies by default unless callers explicitly opt into degraded semantics.
- Python Tree-sitter anchor build and relocation is implemented with Python AST fallback parity coverage.
- The public `TransactionalMemoryKernel` and `BenchmarkRunner` facades remain stable while their internal orchestration is split into smaller collaborators.

## Current implemented capabilities by layer

### Record layer

- Frozen Pydantic records define the durable wire shape for memories, artifacts, transactions, cache entries, retrieval results, verification results, embedding index rows, consolidation results, events, and benchmark records.
- Public records retain `to_json()` / `from_json()` round-trip support for storage and tests.
- Artifact records represent capture occurrences, not unique blob identity. Multiple capture records can point at the same SHA-256-addressed blob.
- `ArtifactIntegrityReport` provides a non-mutating diagnostic view over prepared captures, committed missing blobs, and orphaned blob paths.

### Store layer

- `SqliteMetadataStore`
  - persists committed memory items, lineage edges, transaction records, staged transaction state, and events
  - provides `run_atomically(...)` for grouped metadata mutations
  - treats SQLite as the canonical event ledger
  - supports `export_events_to_jsonl()` as an at-least-once derived JSONL export
  - supports `rebuild_events_jsonl()` to rewrite a deduped log from SQLite source-of-truth rows
- `FilesystemArtifactStore`
  - stores blobs by SHA-256 under `sha256/<digest>`
  - stores per-capture metadata in SQLite
  - tracks prepared, committed, and abandoned capture states
  - provides `audit_integrity()` plus janitor helpers for stale prepared captures and orphaned blobs
- `SqliteCacheStore`
  - stores deterministic cache entries keyed by normalized tool inputs and environment fingerprints
  - performs timezone-safe expiry checks
- `SqliteEmbeddingIndexStore`
  - stores derived embedding rows keyed by `(memory_id, adapter_id)`
  - tracks content digests so stale rows can be refreshed lazily during retrieval

### Service layer

- `TransactionalMemoryKernel`
  - supports nested transactions with ancestor visibility and sibling isolation
  - persists staged state durably across process restarts
  - merges child staged items upward on child commit and only persists memories on root commit
  - remains the public service facade while delegating internally to transaction, validation, artifact, and retrieval collaborators
- `BasicVerifier`
  - compares current dependency fingerprints against stored fingerprints
  - supports parser-agnostic relocation through the configured anchor adapter
- `LexicalRetriever`
  - performs deterministic lexical ranking over committed memory
  - tracks retrieval stats on returned memories
- `EmbeddingRetriever`
  - wraps the same committed-memory query surface as lexical retrieval
  - lazily builds and refreshes embedding rows from auditable memory fields only
  - ranks by cosine similarity, then lexical overlap, then recency
- `RLMRerankingRetriever`
  - wraps lexical retrieval without changing the kernel API
  - reranks candidates through `RLMAdapter`
  - can cache rerank responses against repo and environment fingerprints
- `CommandProcedureValidator`
  - executes local command validators
  - captures stdout and stderr as artifacts
  - supports timeout, output truncation, and environment filtering controls
- `DeterministicConsolidator`
  - supersedes duplicate active memories within the same visibility scope
  - emits lineage edges and consolidation events
  - can generate deterministic summary cards that reference supporting memories

### Adapter layer

- Git repo fingerprint collection is implemented and correctness-hardened for tracked diffs plus untracked file content changes.
- Runtime environment fingerprint collection is implemented.
- Python Tree-sitter anchor construction and relocation is implemented.
- Python AST remains the fallback/parity adapter.
- Deterministic hash embeddings are implemented for local and CI-safe embedding retrieval.
- OpenAI is available as an optional reference adapter for both embeddings and RLM reranking behind the `openai` extra.

### Benchmark layer

- `vtm.benchmarks` defines manifest, case, config, and result records.
- `BenchmarkRunner` remains the public benchmark entrypoint while delegating internally to repo materialization, symbol indexing, suite execution, reporting, and subprocess executor helpers.
- The benchmark runner supports:
  - pinned Git repos or a local synthetic smoke corpus
  - generated SWE-bench Lite manifests backed by local repo caches
  - lexical retrieval evaluation with both `taskish_behavior` and `smoke_identity` slices
  - embedding retrieval evaluation
  - reranked retrieval evaluation through the existing `RLMAdapter` contract
  - drift evaluation through re-verification on commit pairs
  - repo and commit-pair filtering before case generation
  - coding-task packing plus an internal structured subprocess executor wrapper around `--executor-command`
  - multiple deterministic synthetic coding tasks spanning arithmetic, branch, default-handling, path, and collection bugs
  - a checked-in local OpenAI-compatible patcher script for single-shot coding runs
  - official SWE-bench harness aggregation for harness-backed coding tasks
  - coding summaries with pass rate as the primary signal plus resolved-rate, changed-path, and patch diagnostics
- Benchmark outputs live outside the production VTM stores:
  - `manifest.lock.json`
  - `cases.jsonl`
  - `results.jsonl`
  - `summary.json`
  - `summary.md`
  - task packs, workspaces, and executor artifact files

## Current correctness guarantees

- Transaction staging is durable.
- Transaction visibility is isolated.
- Transaction state transitions are atomic in SQLite when metadata and events share the same `SqliteMetadataStore`.
- Verification, procedure validation metadata updates, procedure promotion, and retrieval-stat updates are atomic in SQLite under the same shared-store condition.
- Event export semantics are explicit: SQLite is the source of truth, JSONL export is at-least-once, consumers dedupe by `event_id`, and `rebuild_events_jsonl()` is the repair path.
- Artifact integrity is inspectable through prepared capture state plus non-mutating audits.
- Cache expiry checks do not compare naive and aware datetimes directly.
- Git dirty fingerprints change when tracked diffs change, when an untracked file is created, and when the contents of an already-untracked file change.
- Benchmark case IDs are unique across repo, pair, path, symbol, and retrieval slice, so selected-case counts stay stable even when multiple files define the same symbol name.
- Benchmark summaries report the number of persisted selected cases, and runner validation rejects case/result mismatches before writing outputs.
- Coding task packs now include base/head refs, expected changed paths, target patch digests, memory mode metadata, and richer retrieval context for local executors.
- Coding-task scoring now measures actual changed paths from the workspace diff instead of inferring edits from the task metadata alone.
- SWE-bench Lite preparation creates stable local base/gold refs inside cached repos so the standard VTM coding pipeline can materialize workspaces and retrieval context for external tasks.
- Harness-backed coding runs write `predictions.jsonl`, normalized harness results, and log directories into the benchmark output root.
- Supported store revisions are fixture-backed for metadata, cache, artifact, and embedding stores.
- Repo-wide markdown link and manifest references are checked in tests, and the runtime example remains executable.

## Known correctness and auditability gaps

- JSONL export is still a derived sink, not part of the SQLite commit boundary.
- Cross-store atomicity is still absent. Filesystem artifact writes and SQLite metadata/event writes can still diverge if a failure happens between those boundaries.
- Event atomicity still depends on store topology. The strongest guarantees only apply when `event_store is metadata_store` and both are `SqliteMetadataStore`.
- Procedure validation is locally bounded, not sandboxed.

## Intentionally limited areas

- Embedding retrieval is derived and exact, not an ANN or distributed vector index.
- Consolidation is deterministic and conservative: duplicate superseding plus summary cards are implemented, but learned summarization, TTL forgetting, and cross-scope merging are not.
- Retrieval benchmark queries now include a harder task-oriented slice, but they are still deterministic synthetic prompts rather than a large external benchmark dataset.
- SWE-bench support exists, but the default executor is still a single-shot local patch generator rather than a multi-turn coding agent.
- Benchmark executor support is structured now, but still limited to local subprocess execution with artifact capture. Remote, sandboxed, or provider-specific executors are not built.

## Documentation policy

Documentation is expected to move in lockstep with contract changes. Update `README.md`, affected files under `docs/`, impacted package READMEs, and any relevant ADR in the same change that updates durable behavior.
