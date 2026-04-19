# LiveCodeBench Baselines

LiveCodeBench is available in this repository as an external baseline model benchmark for coding ability. It is not the main VTM memory benchmark and should not be used to reinterpret the paper claim.

## Scope

- LiveCodeBench here is baseline-only
- No VTM memory mode is wired into the baseline runner yet
- Main VTM evidence remains retrieval, drift, and drifted retrieval
- Controlled coding-drift remains the maintained coding benchmark inside the paper story
- SWE-bench Lite remains demoted after empty-patch pilot failures and is not part of the maintained result surface

## Environment

The baseline wrappers use the same maintained OpenRouter environment variables as the rest of `josh-testing`:

```bash
export OPENROUTER_API_KEY=...
export VTM_OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
export VTM_EXECUTION_MODEL=google/gemma-4-31b-it:free
export VTM_RERANK_MODEL=nvidia/nemotron-3-nano-30b-a3b:free
```

## Setup

Clone and install the external LiveCodeBench checkout under `benchmarks/LiveCodeBench`:

```bash
bash scripts/livecodebench/setup_livecodebench.sh
```

That helper uses `uv`, creates `benchmarks/LiveCodeBench/.venv`, and installs the external checkout in place.

## Smoke Command

The minimal baseline smoke run is:

```bash
bash scripts/run_livecodebench_baseline.sh --smoke
```

Default behavior:

- model defaults to `VTM_EXECUTION_MODEL`
- outputs land under `.benchmarks/livecodebench/`
- smoke mode uses `n=1`
- smoke mode disables evaluation
- smoke mode narrows the run to a small fixed date window unless you override it

## Notes

- These wrappers pass OpenRouter credentials through common OpenAI-compatible environment variables for the external LiveCodeBench runner.
- This is baseline model evaluation only. It does not touch VTM verifier semantics, retrieval scoring, drift scoring, or drifted-retrieval scoring.
- Generated launcher bundles and benchmark outputs should stay out of git.
