# VTM

VTM is a repository-memory kernel and benchmark harness scoped to one paper question:

Does verified lexical repository memory help coding agents more than no memory, under fair non-oracle task inputs?

The maintained package boundaries are:

- `vtm`: kernel records, stores, verification, lexical retrieval, procedures, consolidation
- `vtm_dspy`: optional DSPy ReAct and RLM integration on top of the VTM kernel
- `vtm.harness`: task packs, workspaces, executors, scoring
- `vtm_rlm`: thin vendored-RLM bridge and writeback helpers
- `vtm.benchmarks`: manifests, runners, reports, and maintained benchmark orchestration

## Maintained Study Surface

- Maintained inference/execution: OpenRouter only
- Static retrieval: `no_memory`, `naive_lexical`, `verified_lexical`
- Drift verification: `verified_lexical`
- Drifted retrieval: `no_memory`, `naive_lexical`, `verified_lexical`
- Controlled coding-drift: `no_memory`, `naive_lexical`, `verified_lexical`
- Optional secondary ablation: `lexical_rlm_rerank`
- Synthetic smoke tasks: local/dev validation only
- Removed from the maintained surface: SWE-bench Lite, Codex execution, embeddings, terminal-only tracks

External coding prompts no longer expose oracle `expected_changed_paths` or `touched_paths` by default. Those fields remain available for scoring only.

SWE-bench Lite was attempted as an external agent pilot, but the OpenRouter-backed vendored RLM produced empty patches and zero resolved tasks. It is not part of the final maintained benchmark surface or paper claim.

## DSPy Interface

DSPy is the recommended forward-facing agent and programming interface for VTM memory, while VTM itself remains the verified-memory kernel.

- DSPy ReAct plus tools is the preferred tool-using workflow surface
- DSPy RLM is optional long-context reasoning glue, not the maintained local repo-editing executor
- DSPy remains optional and does not change the maintained retrieval, drift, or drifted-retrieval benchmark surface
- Controlled coding-drift remains the small agent-loop benchmark
- LiveCodeBench remains baseline model evaluation only
- SWE-bench Lite remains removed from the maintained result surface after empty-patch pilot failures

## External Baselines

LiveCodeBench support is available for baseline model coding ability checks under OpenRouter, but it is not a maintained VTM memory benchmark.

- LiveCodeBench baseline runs live under `.benchmarks/livecodebench/`
- No VTM memory mode is wired into that baseline runner yet
- The main VTM evidence remains retrieval, drift, and drifted retrieval
- SWE-bench Lite remains removed from the maintained result surface after empty-patch pilot failures

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
uv sync --dev --extra rlm
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

- `benchmarks/manifests/`: checked-in synthetic smoke, controlled coding-drift, and pinned OSS manifests
- `docs/`: maintained benchmark recipes, scope note, audit, harness notes, ADRs
- `src/vtm/`: kernel package plus harness, adapters, services, stores, benchmarks
- `src/vtm_dspy/`: optional DSPy integration layer over the current kernel and workspace tools
- `src/vtm_rlm/`: thin vendored-RLM bridge
- `tests/`: regression coverage for kernel, retrieval, verification, harness, and controlled coding paths

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
```

## Docs

- [docs/final-scope.md](docs/final-scope.md): final paper-facing scope and removed branches
- [docs/final-audit.md](docs/final-audit.md): final maintained surface, rationale, and freeze audit
- [docs/benchmark-recipes.md](docs/benchmark-recipes.md): maintained commands
- [docs/dspy-integration.md](docs/dspy-integration.md): optional DSPy ReAct and RLM integration notes
- [docs/livecodebench-baselines.md](docs/livecodebench-baselines.md): external LiveCodeBench baseline setup and smoke command
- [docs/current-state-audit.md](docs/current-state-audit.md): guarantees and limits
- [docs/harness.md](docs/harness.md): task-pack and executor contract
- [docs/runtime-example.md](docs/runtime-example.md): executable kernel example
- [docs/decisions/README.md](docs/decisions/README.md): kept ADRs
