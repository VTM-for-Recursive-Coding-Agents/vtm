# VTM

VTM is a repository-memory kernel and benchmark harness scoped to one paper question:

Does verified lexical repository memory help coding agents more than no memory, under fair non-oracle task inputs?

The maintained package boundaries are:

- `vtm`: kernel records, stores, verification, lexical retrieval, procedures, consolidation
- `vtm.harness`: task packs, workspaces, executors, scoring
- `vtm_rlm`: thin vendored-RLM bridge and writeback helpers
- `vtm.benchmarks`: manifests, runners, SWE-bench prep/eval helpers

## Maintained Study Surface

- Maintained inference/execution: OpenRouter only
- Retrieval evaluation: `no_memory`, `naive_lexical`, `verified_lexical`
- Drift evaluation: `verified_lexical`
- Coding evaluation: targeted SWE-bench Lite with `no_memory`, `naive_lexical`, or `verified_lexical`
- Synthetic coding smoke: maintained for local/dev validation only
- Optional secondary ablation: `lexical_rlm_rerank`
- Removed from the maintained surface: Codex execution, embeddings, terminal-only tracks

External coding prompts no longer expose oracle `expected_changed_paths` or `touched_paths` by default. Those fields remain available for scoring only.

## OpenRouter Defaults

The maintained inference path uses OpenRouter’s OpenAI-compatible API only.

- Base URL env var: `VTM_OPENROUTER_BASE_URL`
- Default base URL: `https://openrouter.ai/api/v1`
- API key env var: `OPENROUTER_API_KEY`
- Execution model env var: `VTM_EXECUTION_MODEL`
- Rerank model env var: `VTM_RERANK_MODEL`
- Default execution model: `google/gemma-4-31b-it:free`
- Default smoke/dev rerank model: `nvidia/nemotron-3-nano-30b-a3b:free`
- Optional stronger ablation model: `nvidia/nemotron-3-super-120b-a12b:free`

## Environment Setup

Basic dev environment:

```bash
uv sync --dev
```

Full eval environment:

```bash
uv sync --dev --extra rlm --extra bench
export OPENROUTER_API_KEY=...
export VTM_OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
export VTM_EXECUTION_MODEL=google/gemma-4-31b-it:free
export VTM_RERANK_MODEL=nvidia/nemotron-3-nano-30b-a3b:free
```

Vendored-RLM and OpenRouter-backed benchmark tests require the optional `openai` dependency from the `rlm` extra. If you only install the basic dev environment, vendored-RLM tests such as `tests/test_vtm_rlm.py` may skip.

Quick synthetic retrieval run:

```bash
uv run python -m vtm.benchmarks.run \
  --manifest benchmarks/manifests/synthetic-smoke.json \
  --suite retrieval \
  --mode verified_lexical \
  --output .benchmarks/retrieval-verified
```

## Layout

- `benchmarks/manifests/`: checked-in synthetic smoke and pinned OSS manifests
- `docs/`: maintained benchmark recipes, scope note, audit, harness notes, ADRs
- `src/vtm/`: kernel package plus harness, adapters, services, stores, benchmarks
- `src/vtm_rlm/`: thin vendored-RLM bridge
- `tests/`: regression coverage for kernel, retrieval, verification, harness, and SWE-bench paths

## Development

```bash
uv run pytest -q
uv run python -m ruff check .
uv run python -m mypy src
```

CLI entrypoints:

```bash
vtm-bench --help
vtm-bench-compare --help
vtm-bench-matrix --help
vtm-bench-report --help
vtm-prepare-swebench-lite --help
```

## Docs

- [docs/final-scope.md](docs/final-scope.md): final paper-facing scope and removed branches
- [docs/benchmark-recipes.md](docs/benchmark-recipes.md): maintained commands
- [docs/current-state-audit.md](docs/current-state-audit.md): guarantees and limits
- [docs/harness.md](docs/harness.md): task-pack and executor contract
- [docs/runtime-example.md](docs/runtime-example.md): executable kernel example
- [docs/decisions/README.md](docs/decisions/README.md): kept ADRs
