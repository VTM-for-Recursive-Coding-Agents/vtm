# Codebase Guide

This document is the maintainer-oriented map of the repository. Use it when you need to find the owning module for a behavior, choose the right import boundary, or figure out which tests and docs move with a change.

For public contracts and examples, use the source-of-truth docs alongside this guide:

- [`api.md`](api.md): stable kernel-facing API
- [`architecture.md`](architecture.md): system boundaries and data flows
- [`harness.md`](harness.md): task-pack, workspace, executor, and trace contracts
- [`type-system.md`](type-system.md): durable record and enum reference

## Top-level layout

- `src/vtm/`: installable package and the main ownership boundary for runtime behavior
- `docs/`: source-of-truth architecture, API, audit, and ADR material
- `tests/`: regression coverage, fixtures, and docs-parity checks
- `benchmarks/manifests/`: checked-in benchmark corpora and smoke workloads
- `scripts/`: small local utilities that support benchmark workflows

## Public package boundaries

### `vtm`

The root package is kernel-first. Import stable records, stores, and services from [`src/vtm/__init__.py`](../src/vtm/__init__.py).

Use this surface for:

- durable memory, evidence, transaction, retrieval, verification, and consolidation records
- store protocols and built-in SQLite/filesystem implementations
- kernel orchestration via `TransactionalMemoryKernel`

Avoid using the root package for new harness, agent-runtime, or benchmark imports. Those now live in their owning subpackages.

### `vtm.harness`

[`src/vtm/harness/`](../src/vtm/harness/) owns the execution boundary between the kernel and evaluation workflows.

Start here when working on:

- `HarnessTaskPack`, `ExecutorRequest`, `ExecutorResult`, and `TraceManifest`
- local and Docker-backed workspace preparation
- subprocess and native-agent benchmark executors
- changed-path and patch-similarity scoring helpers

### `vtm.agents`

[`src/vtm/agents/`](../src/vtm/agents/) owns the native single-agent runtime.

Start here when working on:

- `TerminalCodingAgent`
- prompt assembly, session/turn/trace records, and compaction
- permission policies and built-in tool registration
- terminal, file, patch, and memory tool implementations

### `vtm.benchmarks`

[`src/vtm/benchmarks/`](../src/vtm/benchmarks/) owns manifest-driven evaluation and reporting.

Start here when working on:

- benchmark manifests, configs, and result records
- suite dispatch for retrieval, drift, coding, and SWE-bench flows
- corpus preparation, repo materialization, and benchmark summaries
- repeated-attempt orchestration and aggregate metrics

## `src/vtm/` module map

### Root record modules

These modules define the durable types most callers interact with directly:

- [`memory_items.py`](../src/vtm/memory_items.py): canonical `MemoryItem`, payload variants, visibility, lineage, and stats
- [`transactions.py`](../src/vtm/transactions.py): `TransactionRecord`
- [`retrieval.py`](../src/vtm/retrieval.py): retrieval request/result contracts
- [`verification.py`](../src/vtm/verification.py): verification and procedure-validation result contracts
- [`artifacts.py`](../src/vtm/artifacts.py): artifact lifecycle records and integrity reports
- [`events.py`](../src/vtm/events.py): durable event rows
- [`evidence.py`](../src/vtm/evidence.py): artifact and memory evidence references
- [`anchors.py`](../src/vtm/anchors.py): code-anchor and relocation contracts
- [`fingerprints.py`](../src/vtm/fingerprints.py): repo, environment, and dependency fingerprints
- [`cache.py`](../src/vtm/cache.py): cache keys and values
- [`embeddings.py`](../src/vtm/embeddings.py): derived embedding-index rows
- [`consolidation.py`](../src/vtm/consolidation.py): consolidation actions and run summaries
- [`enums.py`](../src/vtm/enums.py): cross-cutting enums shared across records
- [`base.py`](../src/vtm/base.py): shared Pydantic base model and schema version
- [`policies.py`](../src/vtm/policies.py), [`ids.py`](../src/vtm/ids.py): smaller utility and identifier helpers

If a change affects one of these durable records, it usually also affects `docs/type-system.md`, API docs, and at least one round-trip or behavior test under `tests/`.

### `services/`

[`src/vtm/services/`](../src/vtm/services/) turns record types, stores, and adapters into the public kernel API.

Key files:

- [`memory_kernel.py`](../src/vtm/services/memory_kernel.py): `MemoryKernel` protocol and `TransactionalMemoryKernel` facade
- [`kernel_transactions.py`](../src/vtm/services/kernel_transactions.py): begin, stage, commit, rollback flow
- [`kernel_mutations.py`](../src/vtm/services/kernel_mutations.py): write-side memory mutation helpers
- [`kernel_retrieval.py`](../src/vtm/services/kernel_retrieval.py): retrieval and expansion flow
- [`kernel_validation.py`](../src/vtm/services/kernel_validation.py): verification, procedure validation, and promotion flow
- [`kernel_artifacts.py`](../src/vtm/services/kernel_artifacts.py): artifact capture and evidence helpers
- [`retriever.py`](../src/vtm/services/retriever.py): lexical retrieval implementation
- [`embedding_retriever.py`](../src/vtm/services/embedding_retriever.py): derived embedding retrieval
- [`reranking_retriever.py`](../src/vtm/services/reranking_retriever.py): optional RLM reranking wrapper
- [`verifier.py`](../src/vtm/services/verifier.py): dependency and anchor-based verification
- [`procedures.py`](../src/vtm/services/procedures.py): command-based procedure validation
- [`consolidator.py`](../src/vtm/services/consolidator.py): deterministic consolidation
- [`fingerprints.py`](../src/vtm/services/fingerprints.py): dependency fingerprint assembly

Use this package when the behavior change is semantic rather than purely structural. Changes here usually need matching updates in `docs/api.md`, `docs/architecture.md`, and behavioral tests such as `tests/test_transactions.py`, `tests/test_verification.py`, `tests/test_retrieval.py`, `tests/test_procedures.py`, or `tests/test_consolidation.py`.

### `stores/`

[`src/vtm/stores/`](../src/vtm/stores/) is the persistence layer.

Key files:

- [`base.py`](../src/vtm/stores/base.py): store protocols
- [`sqlite_store.py`](../src/vtm/stores/sqlite_store.py): metadata store and canonical SQLite-backed event ledger
- [`cache_store.py`](../src/vtm/stores/cache_store.py): deterministic cache persistence
- [`embedding_store.py`](../src/vtm/stores/embedding_store.py): derived embedding index persistence
- [`artifact_store.py`](../src/vtm/stores/artifact_store.py): filesystem artifact blob storage and capture metadata
- [`_sqlite_schema.py`](../src/vtm/stores/_sqlite_schema.py): schema-version definitions
- [`migrations/`](../src/vtm/stores/migrations/): ordered schema migrations for metadata, cache, embedding, and artifact stores

When changing persistence contracts, also check:

- `tests/test_store_migrations.py`
- `tests/test_artifacts.py`
- `tests/test_cache.py`
- fixture SQL under [`tests/fixtures/migrations/`](../tests/fixtures/migrations/)

### `adapters/`

[`src/vtm/adapters/`](../src/vtm/adapters/) contains provider-specific and environment-specific integrations.

Key files:

- [`git.py`](../src/vtm/adapters/git.py): repository fingerprint collection
- [`runtime.py`](../src/vtm/adapters/runtime.py): Python, runtime, and tool fingerprint collection
- [`python_ast.py`](../src/vtm/adapters/python_ast.py), [`tree_sitter.py`](../src/vtm/adapters/tree_sitter.py): syntax trees, code anchors, and relocation
- [`embeddings.py`](../src/vtm/adapters/embeddings.py): embedding adapter contract and deterministic local implementation
- [`rlm.py`](../src/vtm/adapters/rlm.py): provider-neutral reranking contract
- [`agent_model.py`](../src/vtm/adapters/agent_model.py): provider-neutral agent model-turn contract
- [`openai_embedding.py`](../src/vtm/adapters/openai_embedding.py), [`openai_rlm.py`](../src/vtm/adapters/openai_rlm.py), [`openai_chat.py`](../src/vtm/adapters/openai_chat.py), [`openai_agent.py`](../src/vtm/adapters/openai_agent.py): optional OpenAI-compatible implementations

If a change is optional-provider-specific, keep the stable kernel contracts in `vtm` unchanged unless the boundary itself is intentionally moving.

### `harness/`

[`src/vtm/harness/`](../src/vtm/harness/) defines the public execution seam used by benchmarks and the native agent runtime.

Key files:

- [`models.py`](../src/vtm/harness/models.py): task-pack, executor, trace, and context records
- [`workspace.py`](../src/vtm/harness/workspace.py): local workspace preparation and driver lifecycle
- [`workspace_docker.py`](../src/vtm/harness/workspace_docker.py): Docker-backed workspace backend and driver
- [`executors.py`](../src/vtm/harness/executors.py): subprocess and native-agent executor implementations
- [`scoring.py`](../src/vtm/harness/scoring.py): changed-path and patch-similarity metrics

This package is the right place for work that must stay stable even if benchmark orchestration or prompt construction changes.

### `agents/`

[`src/vtm/agents/`](../src/vtm/agents/) is the native runtime package.

Key files:

- [`runtime.py`](../src/vtm/agents/runtime.py): `TerminalCodingAgent` loop and runtime context
- [`models.py`](../src/vtm/agents/models.py): durable request, prompt, turn, tool-call, compaction, and result records
- [`permissions.py`](../src/vtm/agents/permissions.py): interactive and benchmark-autonomous permission policies
- [`tools.py`](../src/vtm/agents/tools.py): public tool-provider entrypoint
- [`tool_terminal.py`](../src/vtm/agents/tool_terminal.py): terminal tool behavior
- [`tool_files.py`](../src/vtm/agents/tool_files.py): file and patch tool behavior
- [`tool_memory.py`](../src/vtm/agents/tool_memory.py): kernel-memory tool behavior
- [`tool_base.py`](../src/vtm/agents/tool_base.py), [`tool_utils.py`](../src/vtm/agents/tool_utils.py): shared tool contracts and helpers
- [`compaction.py`](../src/vtm/agents/compaction.py): context-compaction logic
- [`workspace.py`](../src/vtm/agents/workspace.py): compatibility shim back to `vtm.harness`

If a change affects tool availability or runtime traces, expect matching updates in `docs/harness.md`, the agent README, and tests such as `tests/test_agents.py`, `tests/test_openai_agent.py`, or `tests/test_memory_kernel_runtime.py`.

### `benchmarks/`

[`src/vtm/benchmarks/`](../src/vtm/benchmarks/) is the evaluation orchestration package.

Key files:

- [`models.py`](../src/vtm/benchmarks/models.py): manifests, configs, case records, and run results
- [`runner.py`](../src/vtm/benchmarks/runner.py): public `BenchmarkRunner`
- [`suite_execution.py`](../src/vtm/benchmarks/suite_execution.py): suite dispatch
- [`retrieval_suite.py`](../src/vtm/benchmarks/retrieval_suite.py), [`drift_suite.py`](../src/vtm/benchmarks/drift_suite.py), [`coding_suite.py`](../src/vtm/benchmarks/coding_suite.py): suite-specific logic
- [`kernel_factory.py`](../src/vtm/benchmarks/kernel_factory.py): benchmark-local kernel creation and seeding
- [`reporting.py`](../src/vtm/benchmarks/reporting.py): aggregate metrics and summaries
- [`synthetic.py`](../src/vtm/benchmarks/synthetic.py), [`swebench.py`](../src/vtm/benchmarks/swebench.py), [`swebench_harness.py`](../src/vtm/benchmarks/swebench_harness.py): corpus preparation and SWE-bench integration
- [`repo_materialization.py`](../src/vtm/benchmarks/repo_materialization.py), [`symbol_index.py`](../src/vtm/benchmarks/symbol_index.py): repo prep and indexing helpers
- [`local_patcher.py`](../src/vtm/benchmarks/local_patcher.py): patch generation from typed harness task packs
- [`executor.py`](../src/vtm/benchmarks/executor.py): compatibility shim back to `vtm.harness`

## `docs/` map

Use `docs/` as the source of truth for durable decisions and public behavior:

- [`architecture.md`](architecture.md): package boundaries and end-to-end flows
- [`api.md`](api.md): stable kernel API and import guidance
- [`code-reference.md`](code-reference.md): generated inventory of every Python file and top-level symbol
- [`harness.md`](harness.md): harness contracts and stable artifact layout
- [`type-system.md`](type-system.md): durable records and invariants
- [`current-state-audit.md`](current-state-audit.md): guarantees, correctness gaps, and intentional limits
- [`benchmark-recipes.md`](benchmark-recipes.md): maintained benchmark commands
- [`runtime-example.md`](runtime-example.md): executable end-to-end example
- [`decisions/`](decisions/README.md): ADR index and durable policy decisions

## `tests/` map

The test suite is organized more by behavior than by package.

Useful entry points:

- [`test_types.py`](../tests/test_types.py): import smoke tests and record validation
- [`test_transactions.py`](../tests/test_transactions.py): transaction staging, commit, rollback, and lineage behavior
- [`test_verification.py`](../tests/test_verification.py): dependency and anchor verification
- [`test_retrieval.py`](../tests/test_retrieval.py): lexical and retrieval-surface behavior
- [`test_procedures.py`](../tests/test_procedures.py): procedure validation
- [`test_consolidation.py`](../tests/test_consolidation.py): deterministic consolidation
- [`test_harness.py`](../tests/test_harness.py): harness contracts, workspaces, and compatibility shims
- [`test_agents.py`](../tests/test_agents.py): native runtime, tools, and permission policies
- [`test_benchmarks.py`](../tests/test_benchmarks.py), [`test_benchmark_cli.py`](../tests/test_benchmark_cli.py): benchmark runner and CLI behavior
- [`test_docs_parity.py`](../tests/test_docs_parity.py): executable examples, markdown links, and doc-boundary checks
- [`fixtures/migrations/`](../tests/fixtures/migrations/README.md): schema fixtures used by migration tests

## Common change paths

### Adding or changing a durable kernel record

Touch:

- the owning root module under `src/vtm/`
- `src/vtm/__init__.py` if the type is part of the stable root export surface
- `docs/type-system.md` and possibly `docs/api.md`
- the relevant round-trip or behavior tests under `tests/`

### Changing SQLite-backed persistence or schema

Touch:

- the owning store under `src/vtm/stores/`
- `src/vtm/stores/_sqlite_schema.py`
- the matching file under `src/vtm/stores/migrations/`
- migration fixtures under `tests/fixtures/migrations/`
- `tests/test_store_migrations.py`

### Changing harness or benchmark artifact layout

Touch:

- `src/vtm/harness/models.py`, `workspace.py`, `workspace_docker.py`, or `executors.py`
- `src/vtm/benchmarks/` if result aggregation or reporting changes
- `docs/harness.md`
- `docs/benchmark-recipes.md` or benchmark READMEs if user-facing commands or outputs changed
- `tests/test_harness.py`, `tests/test_benchmarks.py`, and `tests/test_docs_parity.py`

### Changing native-agent tools or permissions

Touch:

- the relevant file under `src/vtm/agents/`
- `src/vtm/adapters/agent_model.py` or OpenAI adapter modules if the model-turn boundary changed
- `src/vtm/agents/README.md` and `docs/harness.md` if the trace or executor contract moved
- `tests/test_agents.py` and related agent/runtime tests

## Reading order for new contributors

1. [`README.md`](../README.md)
2. [`docs/architecture.md`](architecture.md)
3. [`docs/api.md`](api.md)
4. [`src/vtm/README.md`](../src/vtm/README.md)
5. the package README for the area you are changing
6. the owning tests for that behavior
