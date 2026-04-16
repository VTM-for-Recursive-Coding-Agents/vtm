# Current State Audit

This document is the current maintainer-facing snapshot of guarantees, gaps, and explicit limits.

## Current guarantees

- The kernel is typed, frozen at the record layer, and backed by strict Pydantic validation.
- Active transactions persist staged state durably in SQLite and survive process restarts.
- Transaction state changes, verification writeback, procedure validation writeback, procedure promotion, retrieval-stat updates, and event writes are atomic when metadata and events share the same `SqliteMetadataStore`.
- SQLite schema revisions are tracked and future versions are rejected for metadata, cache, artifact, and embedding stores.
- Artifact capture is explicit and auditable through prepared/committed states and `audit_integrity()`.
- Artifact audits now summarize abandoned captures by reason and origin, and `repair_integrity()` applies the safe janitor actions in one pass.
- JSONL event export reconciles complete on-disk lines before appending new rows, so cursor-write failures can resume without replaying already persisted complete events.
- Deterministic lexical retrieval, derived embedding retrieval, optional RLM reranking, and deterministic consolidation are implemented and covered by tests.
- The public benchmark runner now writes typed harness task packs and stable executor artifact layouts.
- Coding benchmarks support repeated attempts with stable per-attempt workspaces, per-attempt artifact roots, `attempts.jsonl`, and aggregate `pass_at_k`/`resolved_at_k` reporting.
- Completed benchmark runs can be compared offline with paired case-level numeric deltas, McNemar-style binary comparisons, bootstrap confidence intervals, and coding `pass_at_k` / `resolved_at_k` summaries derived from `attempts.jsonl`.
- Maintained benchmark matrices can now execute repeated runs across selected modes and materialize baseline comparisons in one pass.
- The checked-in `terminal-smoke` manifest provides a harder local terminal-style benchmark track with explicit `difficulty`, `task_kind`, and optional `retrieval_query`.
- The checked-in `terminal-shell-smoke` manifest provides a shell-command track with explicit `execution_style="shell_command"` and deterministic generated-file tasks.
- `docker_workspace` is implemented as a built-in sandbox backend with normalized container metadata, read-only rootfs defaults, resource limits, and startup logs recorded per attempt.
- `DockerProcedureValidator` reuses the Docker workspace backend to validate procedures against a snapshot of the current repo working tree and records container metadata on the validation result.
- Native-agent shell-command runs enforce `tool_policy="no_file_mutation"` so terminal execution is required while read/search and memory tools remain available.
- Native-agent runs produce stable trace files through the harness boundary and remain benchmark-local, not kernel-persistent.
- Artifact capture and procedure-validation writeback now abandon their capture records best-effort with structured provenance when later event or metadata persistence fails, so cross-store fallout stays inspectable and easier to repair.
- Docs parity checks cover the runtime example, markdown links, and manifest references.

## Known correctness gaps

- JSONL event export is still a derived sink, not part of the SQLite commit boundary.
- Filesystem artifact writes and SQLite metadata/event writes still do not share a single atomic boundary.
- The strongest event guarantees still depend on the shared-store topology.
- `repair_integrity()` only applies safe local cleanup; committed records whose blobs are missing still require operator intervention or regeneration.

## Intentionally limited areas

- The root `vtm` package is kernel-first; benchmark and provider-specific helpers are no longer the primary root import story.
- `vtm.harness` ships local and Docker-backed workspace backends; Docker is the only built-in sandbox backend.
- `CommandProcedureValidator` remains a restricted local-process path; `DockerProcedureValidator` is the only built-in sandboxed procedure-validation backend.
- The native runtime remains single-agent only.
- There is no built-in subagent orchestration or remote sandbox executor.
- Repeated attempts and `pass@k` controls are currently supported only for coding suites.
- Shell-command tasks still live under `suite="coding"`; there is no separate shell-only verification framework.
- Embedding retrieval is derived and exact, not an ANN or distributed vector index.
- Consolidation is conservative and deterministic; learned summarization and forgetting policies remain future work.
- Synthetic coding tasks and SWE-bench Lite support exist, but the harness boundary is still optimized for reproducibility and inspection rather than broad hosted execution.

## Documentation policy

Behavioral or contract changes are expected to update:

- `README.md`
- the relevant source-of-truth doc under `docs/`
- the affected package README
- the relevant ADR when the change is durable
