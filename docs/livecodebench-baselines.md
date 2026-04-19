# LiveCodeBench Baselines

LiveCodeBench is available in this repository as an external baseline model benchmark for coding ability. It is not the main VTM memory benchmark and should not be used to reinterpret the paper claim.

## Scope

- LiveCodeBench here is baseline-only
- No VTM memory mode is wired into the baseline runner yet
- Main VTM evidence remains retrieval, drift verification, drifted retrieval, and controlled coding-drift
- Controlled coding-drift remains the maintained coding benchmark inside the paper story
- SWE-bench Lite remains demoted after empty-patch pilot failures and is not part of the maintained result surface

## What It Measures

LiveCodeBench is useful for baseline model coding ability because it covers tasks such as code generation, self-repair, code execution, and test-output prediction.

That makes it a good external baseline for model capability, but not a direct evaluation of VTM's verified repository memory under drift. It does not replace the main VTM evidence from retrieval, drift verification, drifted retrieval, and controlled coding-drift.

## Environment

The baseline wrappers use the same maintained OpenRouter environment variables as the rest of `josh-testing`:

```bash
export OPENROUTER_API_KEY=...
export VTM_OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
export VTM_EXECUTION_MODEL=google/gemma-4-31b-it:free
export VTM_RERANK_MODEL=nvidia/nemotron-3-nano-30b-a3b:free
```

Single-model runs default to `VTM_EXECUTION_MODEL`.

The maintained OpenRouter baseline matrix is:

- `nvidia/nemotron-3-nano-30b-a3b:free`
- `nvidia/nemotron-3-super-120b-a12b:free`
- `google/gemma-4-31b-it:free`

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
- per-model summaries land under `.benchmarks/paper-tables/livecodebench-baselines/`
- smoke mode uses `n=1`
- smoke mode disables evaluation
- smoke mode narrows the run to a small fixed date window unless you override it
- dry-run is the default and prints the command without calling the model

To actually execute the external benchmark, opt in explicitly:

```bash
bash scripts/run_livecodebench_baseline.sh --smoke --execute
```

## OpenRouter Matrix

Preview the maintained OpenRouter model trio without calling the API:

```bash
bash scripts/run_livecodebench_baseline.sh \
  --model-matrix openrouter-baselines \
  --smoke
```

Run the full matrix only when you are ready to execute it:

```bash
bash scripts/run_livecodebench_baseline.sh \
  --model-matrix openrouter-baselines \
  --execute
```

## Export Results

Raw run metadata stays under `.benchmarks/livecodebench/`. To aggregate those runs into paper-table summaries:

```bash
uv run python scripts/livecodebench/export_results.py \
  --input-root .benchmarks/livecodebench \
  --output-root .benchmarks/paper-tables/livecodebench-baselines
```

That export writes:

- `.benchmarks/paper-tables/livecodebench-baselines/summary.json`
- `.benchmarks/paper-tables/livecodebench-baselines/summary.md`

The repository ignores `.benchmarks/` by default, so raw outputs are not committed unless you override that policy.

## Paper Citation

Use LiveCodeBench in the paper as a baseline model-evaluation reference, not as the main VTM memory result.

Recommended framing:

- LiveCodeBench reports external baseline coding ability for the underlying model.
- VTM's main quantitative evidence remains retrieval, drift verification, drifted retrieval, and controlled coding-drift.
- Controlled coding-drift is the small maintained agent-loop benchmark inside the VTM paper story.

## Notes

- These wrappers pass OpenRouter credentials through common OpenAI-compatible environment variables for the external LiveCodeBench runner.
- This is baseline model evaluation only. It does not touch VTM verifier semantics, retrieval scoring, drift scoring, or drifted-retrieval scoring.
- Generated launcher bundles and benchmark outputs should stay out of git.
