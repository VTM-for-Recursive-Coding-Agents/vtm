# Architecture

VTM is split into a small number of explicit layers:

1. Record layer
   - Frozen Pydantic records define the durable wire and storage format.
   - The canonical records are memory items, artifacts, transactions, retrieval results, verification results, cache entries, embedding index entries, consolidation results, and events.
2. Store layer
   - `SqliteMetadataStore` persists memory items, lineage edges, transactions, staged transaction state, and events.
   - `SqliteMetadataStore` is the canonical event ledger; JSONL output is an explicit at-least-once derived export with cursor state and rebuild support.
   - `FilesystemArtifactStore` persists SHA-256-addressed blobs plus prepared/committed/abandoned artifact capture records.
   - `FilesystemArtifactStore.audit_integrity()` reports prepared captures, committed records with missing blobs, and orphaned blob paths without mutating store state.
   - `SqliteCacheStore` persists deterministic cache entries keyed by tool, normalized args, and environment fingerprints.
   - `SqliteEmbeddingIndexStore` persists derived embedding vectors keyed by `(memory_id, adapter_id)`.
3. Service layer
   - `TransactionalMemoryKernel` remains the public facade and delegates internally to focused transaction, validation, artifact, and retrieval collaborators.
   - `BasicVerifier` updates validity state from dependency fingerprints and optional anchor relocation.
   - `LexicalRetriever` performs deterministic lexical ranking over persisted memory cards.
   - `EmbeddingRetriever` performs derived-vector ranking over the same committed memory set and lazily refreshes the embedding index.
   - `DeterministicConsolidator` performs explicit maintenance passes over committed memory and emits superseding plus summary-card actions.
4. Adapter layer
   - Git fingerprinting and runtime environment collectors are implemented.
   - Python Tree-sitter is the primary anchor adapter; Python AST remains the fallback/parity adapter.
   - Embedding integrations now include a deterministic hash adapter plus an optional OpenAI adapter.
   - RLM integrations use a provider-neutral reranking contract with an optional OpenAI reference adapter.
5. Benchmark layer
   - Benchmark manifests define pinned repo sources, commit pairs, and optional coding tasks.
   - Synthetic smoke corpora are generated locally for deterministic CI-safe retrieval and drift evaluation.
   - SWE-bench Lite tasks can be prepared into generated manifests backed by local repo caches and synthetic gold refs.
   - `BenchmarkRunner` remains the public entrypoint and delegates internally to repo materialization, symbol indexing, suite execution, reporting, and subprocess executor helpers.
   - Retrieval case generation now emits both a harder `taskish_behavior` slice and an explicit `smoke_identity` slice.
   - Benchmark runs can be filtered to selected repos and commit pairs before case generation.
   - Benchmark runs write lockfiles, cases, results, summaries, task packs, workspaces, executor artifacts, and optional SWE-bench prediction and harness artifacts outside the production VTM stores.

## Transaction flow

1. Begin a transaction with a single primary visibility scope.
2. Stage memory items durably in SQLite; children inherit ancestor visibility but siblings remain isolated.
3. Child commit merges staged writes upward without persisting committed memory yet.
4. Root commit persists memory items, lineage edges, transaction state, and corresponding SQLite event rows atomically when the event store is the same `SqliteMetadataStore`.
5. Rollback clears staged state and updates transaction state atomically under the same shared-store condition.
6. The kernel rejects degraded event-store topologies by default; callers must opt in explicitly if they want non-shared event storage semantics.

## Verification and retrieval

- Verification compares a stored dependency fingerprint with the current fingerprint.
- Unchanged dependencies preserve or promote to `verified`.
- Changed code-anchor-backed items become `stale` unless a relocator can produce a new anchor, in which case they become `relocated`.
- Retrieval defaults to `verified` and `relocated` memory and returns explanation metadata plus raw-evidence availability.
- Embedding retrieval derives text only from auditable memory fields: title, summary, tags, validity status, and optional code-anchor path and symbol metadata.
- RLM reranking wraps the existing retriever protocol, leaving `TransactionalMemoryKernel.retrieve(...)` unchanged.
- Verification writeback, procedure validation writeback, procedure promotion, and retrieval-stat updates are grouped into atomic SQLite mutations when metadata and events share the same `SqliteMetadataStore`.
- Artifact-producing flows use explicit prepared -> committed capture transitions so failures leave recoverable artifact state instead of silent cross-store divergence, and store-level integrity audits expose the remaining repair surface.

## Consolidation flow

1. Scan committed memory outside active transaction commit paths.
2. Group candidate duplicates by normalized title, normalized summary, normalized tags, kind, visibility, and dependency fingerprint digest.
3. Keep the newest `verified` or `relocated` memory as canonical.
4. Mark older matching active duplicates as `superseded`, append lineage edges toward the canonical memory, and emit consolidation events.
5. Optionally generate a deterministic `summary_card` over the consolidation group that references the supporting memories through memory evidence refs.

## Benchmark flow

1. Materialize a pinned repo source, either from a remote Git URL or from the synthetic smoke generator.
2. Apply optional repo and commit-pair filters, then check out the base ref for each selected pair.
3. Seed deterministic memory items for eligible non-test Python symbols, attaching both code-anchor and source-snippet evidence.
4. For retrieval, derive both harder task-oriented queries and explicit smoke identity queries for the same committed memory set.
5. For embedding mode, lazily build benchmark-local derived vectors in a dedicated SQLite embedding store.
6. For coding tasks, write a task pack, optionally clone a workspace, run the configured executor command, capture executor artifacts, and score the produced patch.
7. For SWE-bench-backed coding tasks, batch the produced patches into `predictions.jsonl`, invoke the official harness once, and merge normalized harness verdicts back into the benchmark results.
8. Validate that selected cases and persisted results stay aligned, then persist benchmark lockfiles, cases, results, summaries, and any harness artifacts under the chosen output directory without mutating the tracked repository state outside benchmark-local workspaces.
