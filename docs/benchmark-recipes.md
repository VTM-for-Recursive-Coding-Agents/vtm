# Benchmark Recipes

These are the maintained commands for the narrowed paper artifact: OpenRouter-only inference, a verified lexical memory study, no embeddings, and no terminal track.

## Environments

Basic dev environment:

```bash
uv sync --dev
```

Full eval environment:

```bash
uv sync --dev --extra rlm --extra bench
```

The full eval environment is the maintained setup for OpenRouter-backed coding runs, rerank ablations, SWE-bench prep, and paper-table export. If you only install the basic dev environment, vendored-RLM tests that import the optional `openai` package may skip.

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

## Drift

Verified lexical drift run:

```bash
uv run python -m vtm.benchmarks.run \
  --manifest benchmarks/manifests/synthetic-smoke.json \
  --suite drift \
  --mode verified_lexical \
  --output .benchmarks/drift-verified-lexical
```

## Matrix Presets

Retrieval paper preset:

```bash
uv run python -m vtm.benchmarks.matrix \
  --preset synthetic_retrieval \
  --output .benchmarks/matrix-retrieval
```

This maintained preset runs `no_memory`, `naive_lexical`, and `verified_lexical`.

Drift paper preset:

```bash
uv run python -m vtm.benchmarks.matrix \
  --preset synthetic_drift \
  --output .benchmarks/matrix-drift
```

Coding paper preset:

```bash
uv run python -m vtm.benchmarks.matrix \
  --preset synthetic_coding \
  --output .benchmarks/matrix-coding \
  --pair bugfix \
  --execution-model "$VTM_EXECUTION_MODEL"
```

## Synthetic Coding Smoke

No-memory baseline:

```bash
uv run python -m vtm.benchmarks.run \
  --manifest benchmarks/manifests/synthetic-smoke.json \
  --suite coding \
  --mode no_memory \
  --output .benchmarks/coding-no-memory \
  --pair bugfix \
  --execution-model "$VTM_EXECUTION_MODEL"
```

Verified lexical run:

```bash
uv run python -m vtm.benchmarks.run \
  --manifest benchmarks/manifests/synthetic-smoke.json \
  --suite coding \
  --mode verified_lexical \
  --output .benchmarks/coding-verified-lexical \
  --pair bugfix \
  --execution-model "$VTM_EXECUTION_MODEL"
```

Naive lexical ablation:

```bash
uv run python -m vtm.benchmarks.run \
  --manifest benchmarks/manifests/synthetic-smoke.json \
  --suite coding \
  --mode naive_lexical \
  --output .benchmarks/coding-naive-lexical \
  --pair bugfix \
  --execution-model "$VTM_EXECUTION_MODEL"
```

Optional reranked ablation:

```bash
uv run python -m vtm.benchmarks.run \
  --manifest benchmarks/manifests/synthetic-smoke.json \
  --suite coding \
  --mode lexical_rlm_rerank \
  --output .benchmarks/coding-lexical-rlm \
  --pair bugfix \
  --execution-model "$VTM_EXECUTION_MODEL" \
  --rerank-model "$VTM_RERANK_MODEL"
```

## Paper Table Export

Retrieval table export from the maintained three-way matrix:

```bash
uv run python -m vtm.benchmarks.report \
  --retrieval-run .benchmarks/matrix-retrieval/runs/no_memory \
  --retrieval-run .benchmarks/matrix-retrieval/runs/naive_lexical \
  --retrieval-run .benchmarks/matrix-retrieval/runs/verified_lexical \
  --output .benchmarks/paper-tables/retrieval
```

Drift table export:

```bash
uv run python -m vtm.benchmarks.report \
  --drift-run .benchmarks/matrix-drift/runs/verified_lexical \
  --output .benchmarks/paper-tables/drift
```

Coding table export:

```bash
uv run python -m vtm.benchmarks.report \
  --coding-run .benchmarks/matrix-coding/runs/no_memory \
  --coding-run .benchmarks/matrix-coding/runs/naive_lexical \
  --coding-run .benchmarks/matrix-coding/runs/verified_lexical \
  --output .benchmarks/paper-tables/coding
```

Combined draft Markdown across retrieval, drift, and coding:

```bash
uv run python -m vtm.benchmarks.report \
  --retrieval-run .benchmarks/matrix-retrieval/runs/no_memory \
  --retrieval-run .benchmarks/matrix-retrieval/runs/naive_lexical \
  --retrieval-run .benchmarks/matrix-retrieval/runs/verified_lexical \
  --drift-run .benchmarks/matrix-drift/runs/verified_lexical \
  --coding-run .benchmarks/matrix-coding/runs/no_memory \
  --coding-run .benchmarks/matrix-coding/runs/naive_lexical \
  --coding-run .benchmarks/matrix-coding/runs/verified_lexical \
  --output .benchmarks/paper-tables/combined
```

The exporter writes suite CSVs when those suites are provided plus a combined `paper_tables.md` for drafting.

## SWE-bench Lite

Prepare a generated manifest:

```bash
uv sync --extra bench

vtm-prepare-swebench-lite \
  --output-manifest .benchmarks/generated/swebench-lite.json \
  --cache-root .benchmarks/swebench-lite
```

Targeted no-memory run:

```bash
uv run python -m vtm.benchmarks.run \
  --manifest .benchmarks/generated/swebench-lite.json \
  --suite coding \
  --mode no_memory \
  --output .benchmarks/swebench-lite-no-memory \
  --repo astropy__astropy \
  --pair astropy__astropy-14182 \
  --execution-model "$VTM_EXECUTION_MODEL" \
  --swebench-dataset-name princeton-nlp/SWE-bench_Lite
```

Targeted verified lexical run:

```bash
uv run python -m vtm.benchmarks.run \
  --manifest .benchmarks/generated/swebench-lite.json \
  --suite coding \
  --mode verified_lexical \
  --output .benchmarks/swebench-lite-verified-lexical \
  --repo astropy__astropy \
  --pair astropy__astropy-14182 \
  --execution-model "$VTM_EXECUTION_MODEL" \
  --swebench-dataset-name princeton-nlp/SWE-bench_Lite
```

Optional compare:

```bash
uv run python -m vtm.benchmarks.compare \
  --baseline .benchmarks/swebench-lite-no-memory \
  --candidate .benchmarks/swebench-lite-verified-lexical \
  --output .benchmarks/swebench-lite-compare
```

## Comparing Completed Runs

Compare retrieval baselines:

```bash
uv run python -m vtm.benchmarks.compare \
  --baseline .benchmarks/retrieval-no-memory \
  --candidate .benchmarks/retrieval-verified-lexical \
  --output .benchmarks/retrieval-compare
```
