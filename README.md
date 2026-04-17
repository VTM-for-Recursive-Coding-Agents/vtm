# VTM

VTM is a typed memory kernel for coding agents that operate inside mutable repositories.

The repo now has four explicit public surfaces:

- `vtm`: the stable kernel, records, stores, and retrieval/verification services
- `vtm.harness`: typed task-pack, workspace, executor, and scoring contracts
- `vtm_rlm`: the vendored-RLM execution bridge layered on top of VTM memory
- `vtm.benchmarks`: manifest models and the benchmark runner

Compatibility re-exports still exist for some older import paths, but new code should import from the subpackage that actually owns the behavior.

## Stable Today

- frozen Pydantic v2 records for the durable memory and storage layer
- SQLite metadata/events, cache storage, and derived embedding index storage
- filesystem artifact capture with prepared/committed states and integrity audits
- deterministic lexical retrieval with explicit `naive_lexical` and `verified_lexical` benchmark modes, derived embedding retrieval, and optional RLM reranking
- verification, procedure validation, and deterministic consolidation
- typed harness task packs plus local and Docker-backed workspace/executor contracts
- vendored upstream `rlm` as the active recursive execution engine
- retrieval, drift, and coding-task benchmark workflows narrowed to a fair repository-memory study: no memory, naive lexical memory, verified lexical memory, and optional RLM reranking
- checked-in `terminal-smoke` coding tasks for harder local terminal-style evaluation
- checked-in `terminal-shell-smoke` coding tasks for shell-command-driven evaluation
- repeated-attempt coding benchmarks with `attempts.jsonl`, `pass_at_k`, `resolved_at_k`, and `patch_applied_at_k`
- Docker-sandboxed coding attempts with per-attempt container metadata and backend breakdowns
- offline benchmark comparison via `vtm-bench-compare`, including paired case deltas and coding `pass_at_k` comparisons
- preset-driven benchmark matrices via `vtm-bench-matrix`

## Benchmark Scope

The maintained study is now:

- `no_memory`: no retrieval context
- `naive_lexical`: lexical retrieval without validity gating or retrieval-time verification
- `verified_lexical`: lexical retrieval plus retrieval-time verification/relocation before memories are surfaced
- `lexical_rlm_rerank`: optional secondary ablation that reranks the verified lexical candidate set

For external coding tasks such as SWE-bench, oracle `expected_changed_paths` and `touched_paths` are still preserved for scoring, but they are no longer injected into model-visible prompts by default.

## Intentionally Limited

- JSONL export is derived from SQLite, not in the same atomic commit boundary
- filesystem artifact writes and SQLite writes are still separate failure domains, even though failed writeback paths now record abandonment provenance and `repair_integrity()` applies the safe janitor steps
- `CommandProcedureValidator` is still restricted local-process execution; `DockerProcedureValidator` is the only built-in sandboxed procedure-validation backend today
- the default workspace backend is still local; Docker is the only built-in sandbox today
- repeated attempts are only implemented for coding suites
- remote sandbox execution and multi-agent orchestration are still future work

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
from vtm.harness import DockerWorkspaceBackend, HarnessTaskPack, LocalWorkspaceBackend
```

Benchmark imports:

```python
from vtm.benchmarks import BenchmarkRunner
```

Benchmark credibility now has two checked-in tracks:

- `benchmarks/manifests/terminal-smoke.json`: patch-oriented terminal tasks with repeated attempts
- `benchmarks/manifests/terminal-shell-smoke.json`: shell-command tasks intended to be solved from the terminal, optionally under Docker isolation

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
- Navigating the repository: start with [docs/codebase-guide.md](docs/codebase-guide.md).
- Looking for a per-file inventory: use [docs/code-reference.md](docs/code-reference.md).
- Running coding tasks in isolated workspaces: read [docs/harness.md](docs/harness.md) and [`src/vtm/harness/README.md`](src/vtm/harness/README.md).
- Using the vendored-RLM runtime bridge: start in [`src/vtm_rlm/__init__.py`](src/vtm_rlm/__init__.py).
- Running evaluations: use [docs/benchmark-recipes.md](docs/benchmark-recipes.md) as the primary benchmark entrypoint, then [`src/vtm/benchmarks/README.md`](src/vtm/benchmarks/README.md) for package ownership.

## Layout

- `src/vtm/`: kernel package plus harness, benchmarks, adapters, services, and stores
- `src/vtm_rlm/`: vendored-RLM execution bridge and memory writeback helpers
- `docs/`: source-of-truth architecture, API, harness, audit, recipes, and ADR docs
- `tests/`: regression coverage for kernel, harness, vendored-RLM integration, benchmarks, migrations, and docs parity
- `benchmarks/manifests/`: checked-in synthetic and pinned OSS manifests

## Development

Target runtime: Python 3.12.

```bash
uv sync --dev --all-extras
uv run pytest -q
uv run python -m ruff check .
uv run python -m mypy src
```

Installed CLI entrypoints:

```bash
vtm-bench --help
vtm-bench-compare --help
vtm-bench-matrix --help
vtm-prepare-swebench-lite --help
```

Nix workflow:

```bash
nix develop
uv sync --dev --all-extras
```

Package and app entrypoints through the flake:

```bash
nix build .#vtm
nix run .#vtm-bench -- --help
nix shell .#vtm
```

## Docs

- [docs/architecture.md](docs/architecture.md): system boundary and data-flow reference
- [docs/api.md](docs/api.md): kernel API and stable root imports
- [docs/codebase-guide.md](docs/codebase-guide.md): maintainer-oriented repository map and ownership guide
- [docs/code-reference.md](docs/code-reference.md): generated inventory of every Python file and top-level symbols
- [docs/harness.md](docs/harness.md): task packs, executors, and workspace backends
- [docs/current-state-audit.md](docs/current-state-audit.md): guarantees, gaps, and explicit limits
- [docs/final-scope.md](docs/final-scope.md): narrowed benchmark study and maintained evaluation modes
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
