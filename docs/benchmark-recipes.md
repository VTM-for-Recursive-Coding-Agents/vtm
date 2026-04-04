# Benchmark Recipes

These commands are the maintained entrypoints for repeatable local benchmark runs.

## Tonight baseline

Synthetic retrieval without memory:

```bash
uv run python -m vtm.benchmarks.run \
  --manifest benchmarks/manifests/synthetic-smoke.json \
  --suite retrieval \
  --mode no_memory \
  --output .benchmarks/2026-04-03/synth-retrieval-no-memory
```

Synthetic lexical retrieval:

```bash
uv run python -m vtm.benchmarks.run \
  --manifest benchmarks/manifests/synthetic-smoke.json \
  --suite retrieval \
  --mode lexical \
  --output .benchmarks/2026-04-03/synth-retrieval-lexical
```

Synthetic embedding retrieval:

```bash
uv run python -m vtm.benchmarks.run \
  --manifest benchmarks/manifests/synthetic-smoke.json \
  --suite retrieval \
  --mode embedding \
  --output .benchmarks/2026-04-03/synth-retrieval-embedding
```

Synthetic drift:

```bash
uv run python -m vtm.benchmarks.run \
  --manifest benchmarks/manifests/synthetic-smoke.json \
  --suite drift \
  --mode lexical \
  --output .benchmarks/2026-04-03/synth-drift
```

OSS retrieval without memory:

```bash
uv run python -m vtm.benchmarks.run \
  --manifest benchmarks/manifests/oss-python.json \
  --suite retrieval \
  --mode no_memory \
  --output .benchmarks/2026-04-03/oss-retrieval-no-memory
```

OSS lexical retrieval:

```bash
uv run python -m vtm.benchmarks.run \
  --manifest benchmarks/manifests/oss-python.json \
  --suite retrieval \
  --mode lexical \
  --output .benchmarks/2026-04-03/oss-retrieval-lexical
```

## Full synthetic runs

Target one commit pair without truncation:

```bash
uv run python -m vtm.benchmarks.run \
  --manifest benchmarks/manifests/synthetic-smoke.json \
  --suite retrieval \
  --mode lexical \
  --output .benchmarks/synthetic-stable \
  --pair stable
```

Coding benchmark comparisons:

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

Run reranking coverage with a provider-backed adapter:

```bash
uv run python -m vtm.benchmarks.run \
  --manifest benchmarks/manifests/synthetic-smoke.json \
  --suite retrieval \
  --mode lexical_rlm_rerank \
  --output .benchmarks/synthetic-rerank \
  --pair stable \
  --rlm-model "$VTM_OPENAI_MODEL"
```

## Targeted OSS runs

Focus on one repo and commit pair:

```bash
uv run python -m vtm.benchmarks.run \
  --manifest benchmarks/manifests/oss-python.json \
  --suite retrieval \
  --mode lexical \
  --output .benchmarks/oss-click-flag-default \
  --repo click \
  --pair flag_default_sentinel
```

Targeted embedding run:

```bash
uv run python -m vtm.benchmarks.run \
  --manifest benchmarks/manifests/oss-python.json \
  --suite retrieval \
  --mode embedding \
  --output .benchmarks/oss-click-embedding \
  --repo click \
  --pair flag_default_sentinel \
  --embedding-model "$VTM_OPENAI_EMBEDDING_MODEL"
```

## Notes

- Retrieval output now reports both `taskish_behavior` and `smoke_identity` slices.
- Coding task packs now include base/head refs, expected changed paths, target patch digests, memory mode metadata, and richer retrieval context.
- Coding summaries report total tasks, resolved counts, pass rate, resolved rate, changed-path F1, patch similarity, and retrieval/context diagnostics.
- `case_count` in `summary.json` and `summary.md` matches the number of persisted benchmark cases.
- Prefer `--repo` and `--pair` filters over `--max-cases` when you want a stable targeted run.
- Coding-task results remain useful for harness validation, not for external solve-rate claims.

## SWE-bench Lite

For a full Windows and WSL2 setup guide, see [`swebench-lite-windows.md`](swebench-lite-windows.md).

Install the optional benchmark dependencies first:

```bash
uv sync --extra bench
```

Prepare a generated manifest plus local repo caches:

```bash
uv run python -m vtm.benchmarks.prepare_swebench_lite \
  --output-manifest .benchmarks/generated/swebench-lite.json \
  --cache-root .benchmarks/swebench-lite
```

Run a targeted harness-backed coding benchmark:

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

Run the full Lite set with the same local patcher:

```bash
uv run python -m vtm.benchmarks.run \
  --manifest .benchmarks/generated/swebench-lite.json \
  --suite coding \
  --mode lexical \
  --output .benchmarks/swebench-lite-full \
  --executor-command "python $PATCHER_SCRIPT --task-file {task_file} --workspace {workspace}" \
  --swebench-dataset-name princeton-nlp/SWE-bench_Lite
```
