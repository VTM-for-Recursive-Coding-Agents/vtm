# Architecture

VTM is organized around a kernel-first boundary with explicit seams for execution and evaluation.

## Public package boundaries

1. `vtm`
   - Owns durable record types, store protocols and implementations, and the public kernel/service API.
   - This is the stability center of the repository.
2. `vtm.harness`
   - Owns typed task packs, workspace preparation, executor contracts, local executor implementations, and coding-task scoring helpers.
   - This is the execution boundary between the kernel and higher-level evaluation workflows.
3. `vtm.benchmarks`
   - Owns manifests, case generation, suite orchestration, reporting, and SWE-bench integration.
   - It composes the kernel, the harness, and the selected execution engine.
4. `vtm_rlm`
   - Owns the bridge between VTM memory and the vendored upstream `rlm` execution runtime.
   - This package is the preferred execution-integration seam for recursive runs.

## Kernel layers

1. Record layer
   - Frozen Pydantic models define the durable memory, artifact, transaction, retrieval, verification, cache, embedding, and event shapes.
2. Store layer
   - SQLite backs metadata/events, cache storage, and the derived embedding index.
   - Filesystem storage backs SHA-256-addressed artifact blobs plus capture metadata.
3. Service layer
   - `TransactionalMemoryKernel` is the public facade.
   - Verification, retrieval, artifacts, procedure validation, and transaction logic are delegated to smaller collaborators.
4. Adapter layer
   - Git, runtime, syntax/anchor, deterministic embedding, optional OpenAI embedding, and optional reranking adapters stay provider-specific at the edge.

## Harness boundary

`vtm.harness` owns the contracts that must stay stable even when benchmark orchestration or agent prompting changes:

- `HarnessTaskPack`
- `TaskMemoryContextItem`
- `ExecutorRequest`
- `ExecutorResult`
- `WorkspaceDriver` / `WorkspaceBackend`

The attempt-aware coding benchmark contract now depends on a few specific fields:

- `HarnessTaskPack.retrieval_query`
  - optional task-authored retrieval override used when benchmark authors want memory quality measured separately from prompt-query synthesis
- `ExecutorRequest.attempt_index`
  - canonical attempt number, starting at `1`
- `ExecutorRequest.artifact_root`
  - stable per-attempt artifact root used by both vendored-RLM and external executors
- `ExecutorResult.attempt_index`
  - propagated attempt identity for `attempts.jsonl`
- `HarnessTaskPack.execution_style`
  - distinguishes patch-oriented tasks from shell-command tasks while keeping both under `suite="coding"`
- `ExecutorRequest.workspace_backend`
  - records whether the attempt ran in `local_workspace` or `docker_workspace`
- `ExecutorResult.workspace_backend`
  - normalized backend identity emitted into aggregate and per-attempt results
- `ExecutorResult.docker_*`
  - optional sandbox metadata for Docker-backed attempts

The built-in implementations are:

- `LocalWorkspaceBackend`
- `LocalWorkspaceDriver`
- `DockerWorkspaceBackend`
- `DockerWorkspaceDriver`
- `SubprocessBenchmarkExecutor`
- `RLMBenchmarkExecutor`

Docker-backed attempts are prepared with:

- bind-mounted workspace and artifact roots
- `--network none` by default
- `--cap-drop ALL`
- `--security-opt no-new-privileges`
- writable `tmpfs` mounted at `/tmp`

Compatibility shims remain at `vtm.benchmarks.executor`.

That shim is transitional; new code should import from `vtm.harness`.

## Core flows

### Transaction and commit flow

1. Begin a transaction with a single primary visibility scope.
2. Stage memory items durably in SQLite.
3. Child commits merge staged state upward.
4. Root commit persists memories, lineage, transaction state, and event rows atomically when metadata and events share the same `SqliteMetadataStore`.
5. Rollback clears staged state and records the terminal transaction state.

### Verification and retrieval flow

1. A memory stores the dependency fingerprint it was last verified against.
2. Verification compares current fingerprints against the stored fingerprint and optionally relocates code anchors.
3. Retrieval reads committed memory, filters by scope and validity, and returns summary-first or raw evidence depending on the request.
4. Embedding retrieval and reranking are wrappers over the same committed-memory surface; they do not change the kernel API.

### Coding-task execution flow

1. `vtm.benchmarks` selects coding cases from a manifest.
2. Retrieval context is built from repo-scoped kernel memory and converted into a typed `HarnessTaskPack`.
3. The task pack is written once per case under `task-packs/<case-id>.json`.
4. `vtm.harness` prepares one isolated workspace per attempt and runs either:
   - an external command executor, or
   - the vendored-RLM executor
5. Per-attempt workspace and artifact layout is stable:
   - `workspaces/<mode>/<case-id>/attempt-01`
   - `executor-artifacts/<case-id>/attempt-01`
6. The executor writes stable per-attempt artifacts:
   - `command-events.jsonl`
   - `final-git-status.txt`
   - `produced.patch`
   - final verification stdout/stderr files
7. Vendored-RLM runs also emit benchmark-local response and completion artifacts under `rlm/`.
8. Benchmark outputs keep one aggregate row per case in `results.jsonl` and one row per attempt in `attempts.jsonl`.
9. Shell-command tasks still use the same diff and changed-path scoring surface when they intentionally regenerate tracked files.
10. Scoring compares actual changed paths and patch similarity against the expected task contract, then aggregates `pass_at_k`, `resolved_at_k`, and `patch_applied_at_k`.

## Design constraints

- The kernel remains typed and storage-focused.
- Execution concerns live at the harness boundary, not inside the kernel.
- Vendored-RLM prompting and memory writeback live in `vtm_rlm`, not in durable kernel interfaces.
- Benchmarks may evolve quickly, but the task-pack, workspace, and executor seam should stay explicit and inspectable.
- Repeated-attempt orchestration is currently scoped to coding suites; retrieval and drift stay single-attempt.
- Shell-command tasks remain under the coding suite; there is no separate shell-only suite.
- Docker is the only built-in sandbox backend in this pass; remote execution stays out of scope.
