# Benchmark Recipes

These commands are the maintained entrypoints for repeatable local runs.

## Retrieval

Synthetic no-memory baseline:

```bash
uv run python -m vtm.benchmarks.run \
  --manifest benchmarks/manifests/synthetic-smoke.json \
  --suite retrieval \
  --mode no_memory \
  --output .benchmarks/retrieval-no-memory
```

Synthetic lexical retrieval:

```bash
uv run python -m vtm.benchmarks.run \
  --manifest benchmarks/manifests/synthetic-smoke.json \
  --suite retrieval \
  --mode lexical \
  --output .benchmarks/retrieval-lexical
```

Synthetic embedding retrieval:

```bash
uv run python -m vtm.benchmarks.run \
  --manifest benchmarks/manifests/synthetic-smoke.json \
  --suite retrieval \
  --mode embedding \
  --output .benchmarks/retrieval-embedding
```

Target one repo/pair deterministically:

```bash
uv run python -m vtm.benchmarks.run \
  --manifest benchmarks/manifests/oss-python.json \
  --suite retrieval \
  --mode lexical \
  --output .benchmarks/oss-click-flag-default \
  --repo click \
  --pair flag_default_sentinel
```

## Drift

```bash
uv run python -m vtm.benchmarks.run \
  --manifest benchmarks/manifests/synthetic-smoke.json \
  --suite drift \
  --mode lexical \
  --output .benchmarks/drift-lexical
```

## Coding tasks

Dry run that only writes typed task packs and retrieval context:

```bash
uv run python -m vtm.benchmarks.run \
  --manifest benchmarks/manifests/synthetic-smoke.json \
  --suite coding \
  --mode lexical \
  --output .benchmarks/coding-dry-run
```

Harder local terminal-task track:

```bash
uv run python -m vtm.benchmarks.run \
  --manifest benchmarks/manifests/terminal-smoke.json \
  --suite coding \
  --mode lexical \
  --output .benchmarks/terminal-smoke-dry-run
```

Shell-command terminal track:

```bash
export VTM_AGENT_BASE_URL=http://127.0.0.1:8000
export VTM_AGENT_MODEL=qwen3.5-35b-a3b

uv run python -m vtm.benchmarks.run \
  --manifest benchmarks/manifests/terminal-shell-smoke.json \
  --suite coding \
  --mode lexical \
  --output .benchmarks/terminal-shell-smoke \
  --rlm-model-id "$VTM_AGENT_MODEL"
```

Vendored-RLM executor:

```bash
export VTM_AGENT_BASE_URL=http://127.0.0.1:8000
export VTM_AGENT_MODEL=qwen3.5-35b-a3b

uv run python -m vtm.benchmarks.run \
  --manifest benchmarks/manifests/synthetic-smoke.json \
  --suite coding \
  --mode lexical \
  --output .benchmarks/coding-rlm \
  --pair bugfix \
  --rlm-model-id "$VTM_AGENT_MODEL" \
  --workspace-command-timeout-seconds 120 \
  --workspace-max-output-chars 20000
```

Attempt-aware vendored-RLM run on the harder terminal track:

```bash
export VTM_AGENT_BASE_URL=http://127.0.0.1:8000
export VTM_AGENT_MODEL=qwen3.5-35b-a3b

uv run python -m vtm.benchmarks.run \
  --manifest benchmarks/manifests/terminal-smoke.json \
  --suite coding \
  --mode lexical \
  --output .benchmarks/terminal-smoke-rlm \
  --rlm-model-id "$VTM_AGENT_MODEL" \
  --attempts 5 \
  --pass-k 1 \
  --pass-k 5
```

Attempt-aware vendored-RLM run on the shell-command track under Docker:

```bash
export VTM_AGENT_BASE_URL=http://127.0.0.1:8000
export VTM_AGENT_MODEL=qwen3.5-35b-a3b

uv run python -m vtm.benchmarks.run \
  --manifest benchmarks/manifests/terminal-shell-smoke.json \
  --suite coding \
  --mode lexical \
  --output .benchmarks/terminal-shell-rlm \
  --rlm-model-id "$VTM_AGENT_MODEL" \
  --workspace-backend docker_workspace \
  --docker-image python:3.12 \
  --attempts 5 \
  --pass-k 1 \
  --pass-k 5
```

Comparable memory-mode matrix for the terminal-smoke track:

```bash
uv run python -m vtm.benchmarks.run \
  --manifest benchmarks/manifests/terminal-smoke.json \
  --suite coding \
  --mode no_memory \
  --output .benchmarks/terminal-smoke-no-memory

uv run python -m vtm.benchmarks.run \
  --manifest benchmarks/manifests/terminal-smoke.json \
  --suite coding \
  --mode lexical \
  --output .benchmarks/terminal-smoke-lexical

uv run python -m vtm.benchmarks.run \
  --manifest benchmarks/manifests/terminal-smoke.json \
  --suite coding \
  --mode lexical_rlm_rerank \
  --output .benchmarks/terminal-smoke-lexical-rlm

uv run python -m vtm.benchmarks.run \
  --manifest benchmarks/manifests/terminal-smoke.json \
  --suite coding \
  --mode embedding \
  --output .benchmarks/terminal-smoke-embedding
```

Comparable memory-mode matrix for the shell-command track:

```bash
uv run python -m vtm.benchmarks.run \
  --manifest benchmarks/manifests/terminal-shell-smoke.json \
  --suite coding \
  --mode no_memory \
  --output .benchmarks/terminal-shell-no-memory

uv run python -m vtm.benchmarks.run \
  --manifest benchmarks/manifests/terminal-shell-smoke.json \
  --suite coding \
  --mode lexical \
  --output .benchmarks/terminal-shell-lexical

uv run python -m vtm.benchmarks.run \
  --manifest benchmarks/manifests/terminal-shell-smoke.json \
  --suite coding \
  --mode lexical_rlm_rerank \
  --output .benchmarks/terminal-shell-lexical-rlm

uv run python -m vtm.benchmarks.run \
  --manifest benchmarks/manifests/terminal-shell-smoke.json \
  --suite coding \
  --mode embedding \
  --output .benchmarks/terminal-shell-embedding
```

## SWE-bench Lite

Prepare a generated manifest:

```bash
uv sync --extra bench

uv run python -m vtm.benchmarks.prepare_swebench_lite \
  --output-manifest .benchmarks/generated/swebench-lite.json \
  --cache-root .benchmarks/swebench-lite
```

Run a targeted harness-backed slice:

```bash
export VTM_LOCAL_LLM_BASE_URL=http://127.0.0.1:8000
export VTM_LOCAL_LLM_MODEL=qwen3.5-35b-a3b

uv run python -m vtm.benchmarks.run \
  --manifest .benchmarks/generated/swebench-lite.json \
  --suite coding \
  --mode lexical \
  --output .benchmarks/swebench-lite-targeted \
  --repo astropy__astropy \
  --pair astropy__astropy-14182 \
  --rlm-model-id "$VTM_LOCAL_LLM_MODEL" \
  --swebench-dataset-name princeton-nlp/SWE-bench_Lite
```

## Comparing completed runs

Compare two retrieval runs with paired case deltas and confidence intervals:

```bash
uv run python -m vtm.benchmarks.compare \
  --baseline .benchmarks/retrieval-no-memory \
  --candidate .benchmarks/retrieval-lexical \
  --output .benchmarks/retrieval-compare
```

Compare two coding runs and include paired `pass_at_k` / `resolved_at_k` comparisons from `attempts.jsonl`:

```bash
uv run python -m vtm.benchmarks.compare \
  --baseline .benchmarks/terminal-smoke-no-memory \
  --candidate .benchmarks/terminal-smoke-lexical \
  --output .benchmarks/terminal-smoke-compare
```

## Running maintained matrices

Run the maintained terminal-smoke matrix and compare every selected mode against `no_memory`:

```bash
uv run python -m vtm.benchmarks.matrix \
  --preset terminal_smoke \
  --output .benchmarks/terminal-smoke-matrix \
  --mode no_memory \
  --mode lexical \
  --mode embedding \
  --rlm-model-id "$VTM_AGENT_MODEL" \
  --attempts 3 \
  --pass-k 1 \
  --pass-k 3
```

Run the maintained shell-command matrix under Docker:

```bash
uv run python -m vtm.benchmarks.matrix \
  --preset terminal_shell_smoke \
  --output .benchmarks/terminal-shell-matrix \
  --mode no_memory \
  --mode lexical \
  --mode embedding \
  --rlm-model-id "$VTM_AGENT_MODEL" \
  --workspace-backend docker_workspace \
  --docker-image python:3.12 \
  --attempts 3 \
  --pass-k 1 \
  --pass-k 3
```

Include `lexical_rlm_rerank` when an RLM model is configured:

```bash
export VTM_OPENAI_MODEL=gpt-5.4-mini
export VTM_AGENT_MODEL=qwen3.5-35b-a3b

uv run python -m vtm.benchmarks.matrix \
  --preset terminal_smoke \
  --output .benchmarks/terminal-smoke-matrix-rlm \
  --mode no_memory \
  --mode lexical \
  --mode lexical_rlm_rerank \
  --mode embedding \
  --rlm-model-id "$VTM_AGENT_MODEL" \
  --rlm-model "$VTM_OPENAI_MODEL"
```

## Notes

- Retrieval summaries include both `taskish_behavior` and `smoke_identity` slices.
- Coding task packs are written as typed `HarnessTaskPack` JSON under `task-packs/`.
- Repeated-attempt coding runs keep one aggregate row per case in `results.jsonl` and one row per attempt in `attempts.jsonl`.
- Offline comparisons write `comparison.json` plus a human-readable `comparison.md`.
- Matrix runs write one completed benchmark run per mode under `runs/<mode>/` plus baseline comparisons under `comparisons/<baseline>-vs-<mode>/`.
- Attempt-aware workspaces and artifacts live under `workspaces/<mode>/<case-id>/attempt-01` and `executor-artifacts/<case-id>/attempt-01`.
- Case-local executor artifacts always include `command-events.jsonl`, `final-git-status.txt`, `produced.patch`, and final verification stdout/stderr files.
- Vendored-RLM runs additionally emit `response.txt`, `completion.json`, and optional trajectory artifacts under `executor-artifacts/<case-id>/attempt-01/rlm/`.
- Docker-backed attempts require `--workspace-backend docker_workspace` plus `--docker-image`; `--docker-network` defaults to `none`.
- Shell-command tasks still use the coding suite, the standard `test_command` verifier, and diff-based scoring when they regenerate tracked files.
- If `--attempts > 1` and no explicit `--pass-k` values are provided, the runner reports `pass_at_1` and `pass_at_<attempt_count>` by default.
- Use repeated `--attempts` and `--pass-k` runs to compare memory modes under the same vendored-RLM execution engine.
- Prefer `--repo` and `--pair` filters over ad hoc truncation when you want reproducible targeted runs.
