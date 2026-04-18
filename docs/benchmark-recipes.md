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

## Matrix Presets

Retrieval paper preset:

```bash
uv run python -m vtm.benchmarks.matrix \
  --preset synthetic_retrieval \
  --output .benchmarks/matrix-retrieval
```

This maintained preset runs `no_memory`, `naive_lexical`, and `verified_lexical`.

Drifted retrieval paper preset:

```bash
uv run python -m vtm.benchmarks.matrix \
  --preset synthetic_retrieval_drifted \
  --output .benchmarks/matrix-retrieval-drifted
```

This maintained preset runs `no_memory`, `naive_lexical`, and `verified_lexical` with memory seeded on `base_ref` and retrieval executed after checking out `head_ref`.

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

Drifted retrieval table export from the maintained three-way drifted matrix:

```bash
uv run python -m vtm.benchmarks.report \
  --retrieval-run .benchmarks/matrix-retrieval-drifted/runs/no_memory \
  --retrieval-run .benchmarks/matrix-retrieval-drifted/runs/naive_lexical \
  --retrieval-run .benchmarks/matrix-retrieval-drifted/runs/verified_lexical \
  --output .benchmarks/paper-tables/retrieval-drifted
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

uv run --extra bench vtm-prepare-swebench-lite \
  --output-manifest .benchmarks/generated/swebench-lite.json \
  --cache-root .benchmarks/swebench-lite
```

Prepare a robust pilot manifest that skips failed instances and keeps scanning for successful ones:

```bash
uv run --extra bench vtm-prepare-swebench-lite \
  --output-manifest .benchmarks/generated/swebench-lite-pilot.json \
  --cache-root .benchmarks/swebench-lite \
  --max-instances 3 \
  --skip-failed-instances
```

SWE-bench harness-backed coding runs require benchmark extras, a running Docker daemon, and OpenRouter credentials:

```bash
export OPENROUTER_API_KEY=...
export VTM_OPENROUTER_BASE_URL="${VTM_OPENROUTER_BASE_URL:-https://openrouter.ai/api/v1}"
```

Run coding pilot modes one at a time. Avoid matrix mode here because OpenRouter free models can rate limit if the modes start back-to-back.

Optional helper for sequential pilot runs:

```bash
run_swebench_pilot_mode() {
  local mode="$1"
  local cases="$2"
  local output_root="$3"
  uv run --extra bench python -m vtm.benchmarks.run \
    --manifest .benchmarks/generated/swebench-lite-pilot.json \
    --suite coding \
    --mode "$mode" \
    --output "${output_root}/${mode}" \
    --max-cases "$cases" \
    --execution-model "$VTM_EXECUTION_MODEL" \
    --swebench-dataset-name princeton-nlp/SWE-bench_Lite
}
```

Nano smoke/debug workflow only. Use this to prove the harness path works, not to report results:

```bash
export VTM_EXECUTION_MODEL=nvidia/nemotron-3-nano-30b-a3b:free

run_swebench_pilot_mode no_memory 1 .benchmarks/swebench-lite-pilot
run_swebench_pilot_mode naive_lexical 1 .benchmarks/swebench-lite-pilot
run_swebench_pilot_mode verified_lexical 1 .benchmarks/swebench-lite-pilot
```

Nemotron Super is the stronger maintained pilot model. This is still a small agent pilot, not a statistically powered SWE-bench benchmark:

```bash
export VTM_EXECUTION_MODEL=nvidia/nemotron-3-super-120b-a12b:free

# One-case pilot
run_swebench_pilot_mode no_memory 1 .benchmarks/swebench-lite-pilot
run_swebench_pilot_mode naive_lexical 1 .benchmarks/swebench-lite-pilot
run_swebench_pilot_mode verified_lexical 1 .benchmarks/swebench-lite-pilot

# Three-case pilot
run_swebench_pilot_mode no_memory 3 .benchmarks/swebench-lite-pilot-3
run_swebench_pilot_mode naive_lexical 3 .benchmarks/swebench-lite-pilot-3
run_swebench_pilot_mode verified_lexical 3 .benchmarks/swebench-lite-pilot-3
```

Export the final three-row coding table once the `naive_lexical` pilot exists:

```bash
uv run python -m vtm.benchmarks.report \
  --coding-run .benchmarks/swebench-lite-pilot/no_memory \
  --coding-run .benchmarks/swebench-lite-pilot/naive_lexical \
  --coding-run .benchmarks/swebench-lite-pilot/verified_lexical \
  --output .benchmarks/paper-tables/final-swebench-lite-pilot
```

## Comparing Completed Runs

Compare retrieval baselines:

```bash
uv run python -m vtm.benchmarks.compare \
  --baseline .benchmarks/retrieval-no-memory \
  --candidate .benchmarks/retrieval-verified-lexical \
  --output .benchmarks/retrieval-compare
```
