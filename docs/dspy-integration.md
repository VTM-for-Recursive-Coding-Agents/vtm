# DSPy Integration

DSPy is the recommended forward-facing agent and programming interface for VTM, while VTM remains the verified-memory layer underneath it.

## Scope

- `vtm` remains the kernel for memory records, retrieval, verification, and benchmark logic.
- `vtm_dspy` is optional and only loaded when you want a DSPy agent or DSPy RLM integration.
- The final quantitative evidence remains retrieval, drift verification, drifted retrieval, and controlled coding-drift.
- `controlled_coding_drift` remains the small maintained agent-loop benchmark.
- LiveCodeBench remains a baseline model coding benchmark only.
- The LiveCodeBench DSPy plus VTM path is a scaffolded pilot, not the main VTM memory-drift benchmark.
- SWE-bench Lite is still not maintained after the empty-patch pilot failures.

## Install

Base VTM environment:

```bash
uv sync --dev
```

Optional DSPy environment:

```bash
uv sync --dev --extra dspy
```

If you want the existing vendored-RLM benchmark path as well:

```bash
uv sync --dev --extra dspy --extra rlm
```

## OpenRouter Config

The DSPy layer reuses the same OpenRouter settings as the maintained VTM benchmark surface.

```bash
export OPENROUTER_API_KEY=...
export VTM_OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
export VTM_EXECUTION_MODEL=google/gemma-4-31b-it:free
export VTM_RERANK_MODEL=nvidia/nemotron-3-nano-30b-a3b:free
```

For the first paid coding-model run, prefer `qwen/qwen3-coder-next`.

Optional DSPy model override:

```bash
export VTM_DSPY_MODEL=openrouter/nvidia/nemotron-3-super-120b-a12b:free
```

By default, `vtm_dspy` resolves `VTM_DSPY_MODEL` in this order:

1. explicit override passed in code
2. `VTM_DSPY_MODEL`
3. `VTM_EXECUTION_MODEL`, mapped into `openrouter/...`
4. `openrouter/nvidia/nemotron-3-super-120b-a12b:free`

Internally, the runtime maps the stored `openrouter/...` model id onto DSPy's OpenAI-compatible `openai/...` LM naming while keeping `VTM_OPENROUTER_BASE_URL` as the base URL.

## ReAct Surface

`VTMReActCodingAgent` is the minimal tool-using DSPy wrapper.

Available VTM memory tools:

- `search_verified_memory(query, k=5)`
- `search_naive_memory(query, k=5)`
- `expand_memory_evidence(memory_id)`
- `verify_memory(memory_id)`

Available controlled workspace tools when `workspace_root` is supplied:

- `read_file(path)`
- `write_file(path, content)`
- `run_command(command)`
- `git_diff()`

These file and command tools are confined to the explicit workspace root. They do not expose arbitrary filesystem access by default.

## RLM Note

`VTMRLMContextAdapter` is optional long-context glue for DSPy reasoning. It prepares verified memory cards for DSPy RLM-style flows, but it should not be treated as the maintained repo-editing executor.

DSPy RLM's default execution sandbox uses Deno and Pyodide. That makes it a better fit for long-context reasoning and tool mediation than direct local repository editing, unless a custom interpreter is added later.

## Smoke Script

Default dry-run mode:

```bash
uv run python scripts/run_dspy_vtm_smoke.py
```

Dry-run with an explicit workspace root:

```bash
uv run python scripts/run_dspy_vtm_smoke.py --workspace-root .
```

Actual model execution is opt-in:

```bash
uv run --extra dspy python scripts/run_dspy_vtm_smoke.py --run-model
```

## LiveCodeBench DSPy Pilot

`scripts/run_livecodebench_dspy_pilot.py` is a small scaffolded pilot for external LiveCodeBench tasks. It compares a direct OpenRouter baseline, a DSPy baseline without VTM memory, and DSPy with VTM verified-memory tools.

It is deliberately separate from the maintained VTM evidence:

- LiveCodeBench is an external coding benchmark.
- The DSPy plus VTM LiveCodeBench path is a scaffolded pilot only.
- The main VTM evidence remains retrieval, drift verification, drifted retrieval, and controlled coding-drift.

When the pilot runs in `self_repair` mode, the second attempt is a public-feedback repair pass. Each method sees the same previous candidate code and the same visible public-test feedback; only DSPy orchestration and VTM memory access differ.

Dry-run:

```bash
uv run --extra dspy python scripts/run_livecodebench_dspy_pilot.py \
  --method all \
  --scenario self_repair \
  --problem-offset 0 \
  --max-problems 3
```

Larger slices can use `--problem-offset` directly or the `scripts/run_livecodebench_dspy_pilot_batch.sh` helper, which defaults to 25-problem batches.
