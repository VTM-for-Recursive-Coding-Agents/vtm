# VTM

VTM is a typed memory kernel for coding agents that operate inside mutable repositories.

The repo now has four explicit public surfaces:

- `vtm`: the stable kernel, records, stores, and retrieval/verification services
- `vtm.harness`: typed task-pack, workspace, executor, and scoring contracts
- `vtm.agents`: the native single-agent coding runtime
- `vtm.benchmarks`: manifest models and the benchmark runner

Compatibility re-exports still exist for some older import paths, but new code should import from the subpackage that actually owns the behavior.

## Stable Today

- frozen Pydantic v2 records for the durable memory and storage layer
- SQLite metadata/events, cache storage, and derived embedding index storage
- filesystem artifact capture with prepared/committed states and integrity audits
- deterministic lexical retrieval, derived embedding retrieval, and optional RLM reranking
- verification, procedure validation, and deterministic consolidation
- typed harness task packs plus local workspace and executor contracts
- native single-agent terminal runtime with built-in file, patch, terminal, and memory tools
- retrieval, drift, coding-task, native-agent, and SWE-bench Lite benchmark workflows
- checked-in `terminal-smoke` coding tasks for harder local terminal-style evaluation
- repeated-attempt coding benchmarks with `attempts.jsonl`, `pass_at_k`, `resolved_at_k`, and `patch_applied_at_k`

## Intentionally Limited

- JSONL export is derived from SQLite, not in the same atomic commit boundary
- filesystem artifact writes and SQLite writes are still separate failure domains
- procedure validation is bounded but not sandboxed
- the native runtime is still single-agent and local
- the default workspace backend is local only; remote sandbox execution is future work
- repeated attempts are only implemented for coding suites; shell-only task classes are still future work

## Import Boundaries

Kernel-first imports:

```python
from vtm import (
    FilesystemArtifactStore,
    LexicalRetriever,
    SqliteMetadataStore,
    TransactionalMemoryKernel,
)
```

Harness imports:

```python
from vtm.harness import HarnessTaskPack, LocalWorkspaceBackend
```

Benchmark imports:

```python
from vtm.benchmarks import BenchmarkRunner
```

The main benchmark credibility entrypoint is now
`benchmarks/manifests/terminal-smoke.json`, which exercises repeated attempts,
memory retrieval, multi-file fixes, and terminal-style workflows.

## Quick Start

Minimal kernel wiring:

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

For a complete executable example that stages memory, captures artifacts, retrieves, and verifies drift, see [docs/runtime-example.md](docs/runtime-example.md).

## Where To Start

- Building against the kernel: start with [docs/api.md](docs/api.md) and [`src/vtm/__init__.py`](src/vtm/__init__.py).
- Running coding tasks in isolated workspaces: read [docs/harness.md](docs/harness.md) and [`src/vtm/harness/README.md`](src/vtm/harness/README.md).
- Using the native agent runtime: start in [`src/vtm/agents/README.md`](src/vtm/agents/README.md).
- Running evaluations: start in [docs/benchmark-recipes.md](docs/benchmark-recipes.md) and [`src/vtm/benchmarks/README.md`](src/vtm/benchmarks/README.md).

## Layout

- `src/vtm/`: kernel package plus harness, agents, benchmarks, adapters, services, and stores
- `docs/`: source-of-truth architecture, API, harness, audit, recipes, and ADR docs
- `tests/`: regression coverage for kernel, harness, agents, benchmarks, migrations, and docs parity
- `benchmarks/manifests/`: checked-in synthetic and pinned OSS manifests

## Development

Target runtime: Python 3.12.

```bash
uv sync --dev
uv run pytest -q
uv run python -m ruff check .
uv run python -m mypy src
```

Optional extras:

```bash
uv sync --extra openai
uv sync --extra bench
```

## Docs

- [docs/architecture.md](docs/architecture.md): system boundary and data-flow reference
- [docs/api.md](docs/api.md): kernel API and stable root imports
- [docs/harness.md](docs/harness.md): task packs, executors, workspace backends, and traces
- [docs/current-state-audit.md](docs/current-state-audit.md): guarantees, gaps, and explicit limits
- [docs/benchmark-recipes.md](docs/benchmark-recipes.md): repeatable benchmark commands
- [docs/runtime-example.md](docs/runtime-example.md): executable end-to-end kernel example
- [docs/decisions/README.md](docs/decisions/README.md): ADR index

## Documentation Policy

Documentation moves in lockstep with behavior changes.

If a change affects a public contract, file layout, artifact layout, CLI surface, or durable example, update:

- `README.md` if the change is user-facing
- the relevant source-of-truth doc in `docs/`
- the affected package README
- the relevant ADR if the boundary or policy is durable
