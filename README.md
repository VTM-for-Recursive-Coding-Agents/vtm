# VTM

VTM is a typed kernel for verified, transactional memory aimed at coding agents that operate in mutable repositories. The current implementation ships explicit record schemas, SQLite-backed metadata and cache storage, a derived SQLite-backed embedding index, recoverable filesystem artifact capture, lexical retrieval, embedding retrieval, optional RLM reranking, deterministic consolidation, and a benchmark harness for retrieval, drift, and coding-task evaluation.

## Status

This repository currently includes:

- frozen Pydantic v2 models for the public record layer
- SQLite-backed metadata, transactions, staged state, events, cache storage, and embedding index storage
- a filesystem artifact store with SHA-256 blob storage, prepared/committed capture states, integrity audits, and janitor recovery helpers
- concrete Git and runtime environment fingerprint collectors
- Python Tree-sitter anchor construction and relocation with Python AST fallback
- deterministic lexical retrieval, embedding retrieval, and optional RLM reranking
- deterministic duplicate consolidation with superseding and summary-card generation
- explicit procedure promotion and local command-based validation
- schema-tracked SQLite stores with future-version rejection and fixture-backed migration coverage for every supported revision
- decomposed kernel and benchmark internals with stable public facades
- benchmark manifests, a synthetic smoke corpus, pinned OSS corpus manifests, SWE-bench Lite manifest preparation, and a benchmark runner CLI with structured executor and harness artifacts
- pytest coverage for core model invariants and service behavior

Still intentionally limited:

- JSONL export remains at-least-once, not exactly-once atomic with SQLite
- filesystem artifact writes and SQLite metadata/events still do not share a single atomic boundary
- procedure validation is bounded but still not sandboxed
- coding evaluation remains single-shot and local; there is no multi-turn coding agent loop or pass@k orchestration

## Layout

- `src/vtm/`: public package modules, stores, services, adapters, and benchmark support
- `docs/`: architecture, type-system, API notes, current-state audit, and ADRs
- `tests/`: round-trip, storage, migration, transaction, verification, retrieval, consolidation, benchmark, and docs-parity tests
- `benchmarks/manifests/`: synthetic and pinned OSS benchmark manifests

## Development

Target runtime: Python 3.12. The current local environment may be newer; the code is kept 3.12-compatible.

Typical commands:

```bash
uv sync --dev
uv run pytest -q
uv run python -m ruff check .
uv run python -m mypy src
```

To enable the optional OpenAI adapters:

```bash
uv sync --extra openai
export OPENAI_API_KEY=...
export VTM_OPENAI_MODEL=...
export VTM_OPENAI_EMBEDDING_MODEL=...
```

To enable the optional SWE-bench Lite preparation and official harness integration:

```bash
uv sync --extra bench
```

Documentation policy: update `README.md`, affected files under `docs/`, impacted package READMEs, and any relevant ADR in the same change that updates a durable behavior or public contract. Schema compatibility policy lives in [`docs/decisions/0005-schema-compatibility-policy.md`](docs/decisions/0005-schema-compatibility-policy.md).

## Runtime Example

See [`docs/runtime-example.md`](docs/runtime-example.md) for a minimal end-to-end flow that:

- collects repo and env fingerprints
- captures a tool artifact
- audits artifact integrity
- builds artifact evidence
- stages and commits a memory
- retrieves it with summary-first behavior
- re-verifies it after repository state changes

Procedure-specific API notes live in [`docs/api.md`](docs/api.md). Current implementation status and remaining correctness gaps live in [`docs/current-state-audit.md`](docs/current-state-audit.md). Event/export and artifact lifecycle guarantees live in [`docs/decisions/0002-event-and-artifact-contracts.md`](docs/decisions/0002-event-and-artifact-contracts.md). Repeatable benchmark recipes live in [`docs/benchmark-recipes.md`](docs/benchmark-recipes.md).

## Benchmarks

Synthetic smoke retrieval with lexical scoring:

```bash
uv run python -m vtm.benchmarks.run \
  --manifest benchmarks/manifests/synthetic-smoke.json \
  --suite retrieval \
  --mode lexical \
  --output .benchmarks/smoke-retrieval \
  --top-k 5
```

Pair-filtered synthetic drift:

```bash
uv run python -m vtm.benchmarks.run \
  --manifest benchmarks/manifests/synthetic-smoke.json \
  --suite drift \
  --mode lexical \
  --output .benchmarks/smoke-drift-stable \
  --pair stable
```

Synthetic smoke retrieval with deterministic embeddings:

```bash
uv run python -m vtm.benchmarks.run \
  --manifest benchmarks/manifests/synthetic-smoke.json \
  --suite retrieval \
  --mode embedding \
  --output .benchmarks/smoke-embedding \
  --top-k 5
```

Lexical plus OpenAI reranking:

```bash
uv run python -m vtm.benchmarks.run \
  --manifest benchmarks/manifests/oss-python.json \
  --suite retrieval \
  --mode lexical_rlm_rerank \
  --output .benchmarks/oss-rerank \
  --top-k 5 \
  --repo click \
  --pair flag_default_sentinel \
  --rlm-model "$VTM_OPENAI_MODEL"
```

Embedding retrieval with an OpenAI embedding model:

```bash
uv run python -m vtm.benchmarks.run \
  --manifest benchmarks/manifests/oss-python.json \
  --suite retrieval \
  --mode embedding \
  --output .benchmarks/oss-embedding \
  --top-k 5 \
  --repo click \
  --pair flag_default_sentinel \
  --embedding-model "$VTM_OPENAI_EMBEDDING_MODEL"
```

Coding-task benchmark dry run:

```bash
uv run python -m vtm.benchmarks.run \
  --manifest benchmarks/manifests/synthetic-smoke.json \
  --suite coding \
  --mode lexical \
  --output .benchmarks/coding-dry-run
```

Coding benchmark comparison runs:

```bash
uv run python -m vtm.benchmarks.run \
  --manifest benchmarks/manifests/synthetic-smoke.json \
  --suite coding \
  --mode no_memory \
  --output .benchmarks/coding-no-memory

uv run python -m vtm.benchmarks.run \
  --manifest benchmarks/manifests/synthetic-smoke.json \
  --suite coding \
  --mode lexical \
  --output .benchmarks/coding-lexical

uv run python -m vtm.benchmarks.run \
  --manifest benchmarks/manifests/synthetic-smoke.json \
  --suite coding \
  --mode embedding \
  --output .benchmarks/coding-embedding
```

Prepare a SWE-bench Lite manifest backed by local repo caches:

```bash
uv run python -m vtm.benchmarks.prepare_swebench_lite \
  --output-manifest .benchmarks/generated/swebench-lite.json \
  --cache-root .benchmarks/swebench-lite
```

Run a targeted SWE-bench Lite coding benchmark with a local OpenAI-compatible patcher:

```bash
export VTM_LOCAL_LLM_BASE_URL=http://127.0.0.1:8000
export VTM_LOCAL_LLM_MODEL=qwen3.5-35b-a3b
export PATCHER_SCRIPT="$PWD/scripts/vtm_local_patcher.py"

uv run python -m vtm.benchmarks.run \
  --manifest .benchmarks/generated/swebench-lite.json \
  --suite coding \
  --mode lexical \
  --output .benchmarks/swebench-lite-qwen-q4 \
  --repo astropy__astropy \
  --pair astropy__astropy-14182 \
  --executor-command "python $PATCHER_SCRIPT --task-file {task_file} --workspace {workspace}" \
  --swebench-dataset-name princeton-nlp/SWE-bench_Lite
```

Coding-task execution is optional. If you want the runner to invoke an external coding agent or script, pass `--executor-command` with `{task_file}` and `{workspace}` placeholders. The runner now writes benchmark-local executor stdout, stderr, and produced patch artifacts under the chosen output directory.

Retrieval runs now emit both `taskish_behavior` and `smoke_identity` slices in `summary.json` / `summary.md`. Use `--repo` and `--pair` filters to keep targeted runs reproducible without depending on unsafe truncation.

Coding task packs now include base/head refs, expected changed paths, target patch digests, memory mode metadata, richer retrieval context, and SWE-bench dataset metadata when applicable. Harness-backed coding runs additionally write `predictions.jsonl`, normalized SWE-bench harness results, and harness logs alongside the standard summaries.
