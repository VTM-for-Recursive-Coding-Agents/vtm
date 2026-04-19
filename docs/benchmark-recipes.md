# Benchmark Recipes

These are the maintained commands for the narrowed paper artifact: OpenRouter-only inference, a verified lexical memory study, DSPy as the forward-facing optional agent integration, and LiveCodeBench as external baseline-model infrastructure.

## Environments

Basic dev environment:

```bash
uv sync --dev
```

Full eval environment:

```bash
uv sync --dev --extra rlm
```

The full eval environment is the maintained setup for OpenRouter-backed coding runs, rerank ablations, and paper-table export. If you only install the basic dev environment, vendored-RLM tests that import the optional `openai` package may skip.

Optional DSPy environment:

```bash
uv sync --dev --extra dspy
```

## OpenRouter Setup

All maintained inference and coding execution paths use OpenRouter's OpenAI-compatible API only.

```bash
export OPENROUTER_API_KEY=...
export VTM_OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
export VTM_EXECUTION_MODEL=google/gemma-4-31b-it:free
export VTM_RERANK_MODEL=nvidia/nemotron-3-nano-30b-a3b:free
```

Recommended models:

- smoke or cheap rerank runs: `nvidia/nemotron-3-nano-30b-a3b:free`
- main execution runs: `google/gemma-4-31b-it:free`
- optional stronger ablation: `nvidia/nemotron-3-super-120b-a12b:free`

## LiveCodeBench Baseline

LiveCodeBench is available here as an external baseline model benchmark only. It is not the main VTM memory-drift benchmark, and it does not change the maintained retrieval, drift, drifted-retrieval, or controlled coding-drift evidence.

Set up the external benchmark checkout:

```bash
bash scripts/livecodebench/setup_livecodebench.sh
```

OpenRouter-backed smoke command:

```bash
export OPENROUTER_API_KEY=...
export VTM_OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
export VTM_EXECUTION_MODEL=google/gemma-4-31b-it:free

bash scripts/run_livecodebench_baseline.sh --smoke
```

Opt in to actual execution explicitly:

```bash
bash scripts/run_livecodebench_baseline.sh --smoke --execute
```

Preview the maintained OpenRouter baseline trio without calling the API:

```bash
bash scripts/run_livecodebench_baseline.sh \
  --model-matrix openrouter-baselines \
  --smoke
```

This baseline runner is memory-free by design. No VTM retrieval or verifier path is involved.

## DSPy Smoke

Dry-run smoke for the optional DSPy integration:

```bash
uv run python scripts/run_dspy_vtm_smoke.py --workspace-root .
```

Opt-in model execution:

```bash
uv run --extra dspy python scripts/run_dspy_vtm_smoke.py --workspace-root . --run-model
```

DSPy is the recommended forward-facing agent interface for VTM memory, and it remains optional; it does not change the maintained retrieval, drift, drifted-retrieval, or controlled coding-drift scoring surfaces.

## Retrieval

No-memory baseline:

```bash
uv run python -m vtm.benchmarks.run \
  --manifest benchmarks/manifests/synthetic-smoke.json \
  --suite retrieval \
  --mode no_memory \
  --output .benchmarks/retrieval-no-memory
```

Verified lexical retrieval:

```bash
uv run python -m vtm.benchmarks.run \
  --manifest benchmarks/manifests/synthetic-smoke.json \
  --suite retrieval \
  --mode verified_lexical \
  --output .benchmarks/retrieval-verified-lexical
```

Naive lexical ablation:

```bash
uv run python -m vtm.benchmarks.run \
  --manifest benchmarks/manifests/synthetic-smoke.json \
  --suite retrieval \
  --mode naive_lexical \
  --output .benchmarks/retrieval-naive-lexical
```

Drifted verified lexical retrieval seeded on `base_ref` and queried on `head_ref`:

```bash
uv run python -m vtm.benchmarks.run \
  --manifest benchmarks/manifests/synthetic-smoke.json \
  --suite retrieval \
  --mode verified_lexical \
  --seed-on-base-query-on-head \
  --output .benchmarks/retrieval-drifted-verified-lexical
```

Drifted naive lexical ablation seeded on `base_ref` and queried on `head_ref`:

```bash
uv run python -m vtm.benchmarks.run \
  --manifest benchmarks/manifests/synthetic-smoke.json \
  --suite retrieval \
  --mode naive_lexical \
  --seed-on-base-query-on-head \
  --output .benchmarks/retrieval-drifted-naive-lexical
```

Optional reranked ablation:

```bash
uv run python -m vtm.benchmarks.run \
  --manifest benchmarks/manifests/synthetic-smoke.json \
  --suite retrieval \
  --mode lexical_rlm_rerank \
  --output .benchmarks/retrieval-lexical-rlm \
  --rerank-model "$VTM_RERANK_MODEL"
```

Target one pinned OSS pair:

```bash
uv run python -m vtm.benchmarks.run \
  --manifest benchmarks/manifests/oss-python.json \
  --suite retrieval \
  --mode verified_lexical \
  --output .benchmarks/oss-click-flag-default \
  --repo click \
  --pair flag_default_sentinel
```

Target one pinned OSS pair in drifted retrieval mode:

```bash
uv run python -m vtm.benchmarks.run \
  --manifest benchmarks/manifests/oss-python.json \
  --suite retrieval \
  --mode no_memory \
  --seed-on-base-query-on-head \
  --output .benchmarks/oss-click-flag-default-drifted-no-memory \
  --repo click \
  --pair flag_default_sentinel
```

```bash
uv run python -m vtm.benchmarks.run \
  --manifest benchmarks/manifests/oss-python.json \
  --suite retrieval \
  --mode naive_lexical \
  --seed-on-base-query-on-head \
  --output .benchmarks/oss-click-flag-default-drifted-naive \
  --repo click \
  --pair flag_default_sentinel
```

```bash
uv run python -m vtm.benchmarks.run \
  --manifest benchmarks/manifests/oss-python.json \
  --suite retrieval \
  --mode verified_lexical \
  --seed-on-base-query-on-head \
  --output .benchmarks/oss-click-flag-default-drifted-verified \
  --repo click \
  --pair flag_default_sentinel
```

Export that drifted OSS comparison:

```bash
uv run python -m vtm.benchmarks.report \
  --retrieval-run .benchmarks/oss-click-flag-default-drifted-no-memory \
  --retrieval-run .benchmarks/oss-click-flag-default-drifted-naive \
  --retrieval-run .benchmarks/oss-click-flag-default-drifted-verified \
  --output .benchmarks/paper-tables/oss-click-flag-default-drifted
```

Swap `--repo` / `--pair` to `attrs` / `frozen_setattr_support` or `rich` / `cells_defensive_fix` for the other maintained OSS pairs.

## Drift

Verified lexical drift run:

```bash
uv run python -m vtm.benchmarks.run \
  --manifest benchmarks/manifests/synthetic-smoke.json \
  --suite drift \
  --mode verified_lexical \
  --output .benchmarks/drift-verified-lexical
```

## Final Paper Commands

Static retrieval:

```bash
uv run python -m vtm.benchmarks.matrix \
  --preset synthetic_retrieval \
  --output .benchmarks/matrix-retrieval
```

Drift verification:

```bash
uv run python -m vtm.benchmarks.matrix \
  --preset synthetic_drift \
  --output .benchmarks/matrix-drift
```

Drifted retrieval:

```bash
uv run python -m vtm.benchmarks.matrix \
  --preset synthetic_retrieval_drifted \
  --output .benchmarks/matrix-retrieval-drifted
```

Controlled coding-drift:

Nano smoke run:

```bash
export VTM_EXECUTION_MODEL=nvidia/nemotron-3-nano-30b-a3b:free

uv run --extra rlm python -m vtm.benchmarks.matrix \
  --preset controlled_coding_drift \
  --output .benchmarks/controlled-coding-drift-nano \
  --execution-model "$VTM_EXECUTION_MODEL"
```

Nemotron Super run:

```bash
export VTM_EXECUTION_MODEL=nvidia/nemotron-3-super-120b-a12b:free

uv run --extra rlm python -m vtm.benchmarks.matrix \
  --preset controlled_coding_drift \
  --output .benchmarks/controlled-coding-drift-super \
  --execution-model "$VTM_EXECUTION_MODEL"
```

Report export:

```bash
uv run python -m vtm.benchmarks.report \
  --coding-run .benchmarks/controlled-coding-drift-super/runs/no_memory \
  --coding-run .benchmarks/controlled-coding-drift-super/runs/naive_lexical \
  --coding-run .benchmarks/controlled-coding-drift-super/runs/verified_lexical \
  --output .benchmarks/paper-tables/controlled-coding-drift-super
```

LiveCodeBench metadata export:

```bash
uv run python scripts/livecodebench/export_results.py \
  --input-root .benchmarks/livecodebench \
  --output-root .benchmarks/paper-tables/livecodebench-baselines
```

## Paper Table Export

Final static retrieval table:

```bash
uv run python -m vtm.benchmarks.report \
  --retrieval-run .benchmarks/matrix-retrieval/runs/no_memory \
  --retrieval-run .benchmarks/matrix-retrieval/runs/naive_lexical \
  --retrieval-run .benchmarks/matrix-retrieval/runs/verified_lexical \
  --output .benchmarks/paper-tables/retrieval
```

Final drifted retrieval table:

```bash
uv run python -m vtm.benchmarks.report \
  --retrieval-run .benchmarks/matrix-retrieval-drifted/runs/no_memory \
  --retrieval-run .benchmarks/matrix-retrieval-drifted/runs/naive_lexical \
  --retrieval-run .benchmarks/matrix-retrieval-drifted/runs/verified_lexical \
  --output .benchmarks/paper-tables/retrieval-drifted
```

Final drift table:

```bash
uv run python -m vtm.benchmarks.report \
  --drift-run .benchmarks/matrix-drift/runs/verified_lexical \
  --output .benchmarks/paper-tables/drift
```

Final controlled coding-drift table:

```bash
uv run python -m vtm.benchmarks.report \
  --coding-run .benchmarks/controlled-coding-drift-super/runs/no_memory \
  --coding-run .benchmarks/controlled-coding-drift-super/runs/naive_lexical \
  --coding-run .benchmarks/controlled-coding-drift-super/runs/verified_lexical \
  --output .benchmarks/paper-tables/controlled-coding-drift-super
```

Combined paper tables:

```bash
uv run python -m vtm.benchmarks.report \
  --retrieval-run .benchmarks/matrix-retrieval/runs/no_memory \
  --retrieval-run .benchmarks/matrix-retrieval/runs/naive_lexical \
  --retrieval-run .benchmarks/matrix-retrieval/runs/verified_lexical \
  --drift-run .benchmarks/matrix-drift/runs/verified_lexical \
  --coding-run .benchmarks/controlled-coding-drift-super/runs/no_memory \
  --coding-run .benchmarks/controlled-coding-drift-super/runs/naive_lexical \
  --coding-run .benchmarks/controlled-coding-drift-super/runs/verified_lexical \
  --output .benchmarks/paper-tables/combined
```

The exporter writes suite CSVs when those suites are provided plus a combined `paper_tables.md` for drafting.

## Comparing Completed Runs

Compare retrieval baselines:

```bash
uv run python -m vtm.benchmarks.compare \
  --baseline .benchmarks/retrieval-no-memory \
  --candidate .benchmarks/retrieval-verified-lexical \
  --output .benchmarks/retrieval-compare
```

## Scope note

SWE-bench Lite was removed from the maintained result surface after the external pilot produced empty patches and no resolved tasks. Future external agent benchmarks can be explored later, but they are not final paper results.
