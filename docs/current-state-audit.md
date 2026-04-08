# Current State Audit

This document is the current maintainer-facing snapshot of guarantees, gaps, and explicit limits.

## Current guarantees

- The kernel is typed, frozen at the record layer, and backed by strict Pydantic validation.
- Active transactions persist staged state durably in SQLite and survive process restarts.
- Transaction state changes, verification writeback, procedure validation writeback, procedure promotion, retrieval-stat updates, and event writes are atomic when metadata and events share the same `SqliteMetadataStore`.
- SQLite schema revisions are tracked and future versions are rejected for metadata, cache, artifact, and embedding stores.
- Artifact capture is explicit and auditable through prepared/committed states and `audit_integrity()`.
- Deterministic lexical retrieval, derived embedding retrieval, optional RLM reranking, and deterministic consolidation are implemented and covered by tests.
- The public benchmark runner now writes typed harness task packs and stable executor artifact layouts.
- Coding benchmarks support repeated attempts with stable per-attempt workspaces, per-attempt artifact roots, `attempts.jsonl`, and aggregate `pass_at_k`/`resolved_at_k` reporting.
- The checked-in `terminal-smoke` manifest provides a harder local terminal-style benchmark track with explicit `difficulty`, `task_kind`, and optional `retrieval_query`.
- Native-agent runs produce stable trace files through the harness boundary and remain benchmark-local, not kernel-persistent.
- Docs parity checks cover the runtime example, markdown links, and manifest references.

## Known correctness gaps

- JSONL event export is still a derived sink, not part of the SQLite commit boundary.
- Filesystem artifact writes and SQLite metadata/event writes still do not share a single atomic boundary.
- The strongest event guarantees still depend on the shared-store topology.
- Procedure validation is locally bounded, not sandboxed.

## Intentionally limited areas

- The root `vtm` package is kernel-first; benchmark and provider-specific helpers are no longer the primary root import story.
- `vtm.harness` currently ships only a local workspace backend and local executors.
- The native runtime remains single-agent only.
- There is no built-in subagent orchestration or remote sandbox executor.
- Repeated attempts and `pass@k` controls are currently supported only for coding suites.
- Coding benchmarks are still patch-and-verify oriented; shell-only task classes are not implemented yet.
- Embedding retrieval is derived and exact, not an ANN or distributed vector index.
- Consolidation is conservative and deterministic; learned summarization and forgetting policies remain future work.
- Synthetic coding tasks and SWE-bench Lite support exist, but the harness boundary is still optimized for reproducibility and inspection rather than broad hosted execution.

## Documentation policy

Behavioral or contract changes are expected to update:

- `README.md`
- the relevant source-of-truth doc under `docs/`
- the affected package README
- the relevant ADR when the change is durable
