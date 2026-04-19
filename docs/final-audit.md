# Final Audit

## Final maintained benchmark surface

1. static retrieval
   - `no_memory`
   - `naive_lexical`
   - `verified_lexical`
2. drift verification
   - `verified_lexical`
3. drifted retrieval
   - `no_memory`
   - `naive_lexical`
   - `verified_lexical`
4. controlled coding-drift
   - `no_memory`
   - `naive_lexical`
   - `verified_lexical`
5. optional secondary ablation
   - `lexical_rlm_rerank`

## Removed SWE-bench rationale

SWE-bench Lite was attempted as an external agent pilot, but the OpenRouter-backed RLM produced empty patches and no resolved tasks. We removed it from the maintained result surface and use controlled coding-drift plus retrieval/drifted retrieval for the final paper.

## Final result interpretation

- The main paper claim is memory correctness and usefulness under repository drift.
- Retrieval, drift verification, and drifted retrieval remain the primary evaluation layers.
- The maintained coding benchmark is `controlled_coding_drift`, not SWE-bench Lite.
- OpenRouter-backed vendored RLM is the maintained execution path.
- DSPy is an optional forward-facing agent interface, not part of the frozen quantitative benchmark surface.
- LiveCodeBench is external baseline-model infrastructure, not part of the frozen quantitative memory result.

## Final run commands

```bash
export OPENROUTER_API_KEY=...
export VTM_OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
export VTM_EXECUTION_MODEL=nvidia/nemotron-3-nano-30b-a3b:free
export VTM_RERANK_MODEL=nvidia/nemotron-3-nano-30b-a3b:free

uv run python -m vtm.benchmarks.matrix \
  --preset synthetic_retrieval \
  --output .benchmarks/matrix-retrieval

uv run python -m vtm.benchmarks.matrix \
  --preset synthetic_drift \
  --output .benchmarks/matrix-drift

uv run python -m vtm.benchmarks.matrix \
  --preset synthetic_retrieval_drifted \
  --output .benchmarks/matrix-retrieval-drifted

uv run --extra rlm python -m vtm.benchmarks.matrix \
  --preset controlled_coding_drift \
  --output .benchmarks/controlled-coding-drift-nano \
  --execution-model "$VTM_EXECUTION_MODEL"

export VTM_EXECUTION_MODEL=nvidia/nemotron-3-super-120b-a12b:free

uv run --extra rlm python -m vtm.benchmarks.matrix \
  --preset controlled_coding_drift \
  --output .benchmarks/controlled-coding-drift-super \
  --execution-model "$VTM_EXECUTION_MODEL"

uv run python -m vtm.benchmarks.report \
  --coding-run .benchmarks/controlled-coding-drift-super/runs/no_memory \
  --coding-run .benchmarks/controlled-coding-drift-super/runs/naive_lexical \
  --coding-run .benchmarks/controlled-coding-drift-super/runs/verified_lexical \
  --output .benchmarks/paper-tables/controlled-coding-drift-super
```

## Tests run

- `uv run --extra dev --extra rlm python -m pytest -q tests/test_benchmark_cli.py tests/test_benchmark_report.py tests/test_benchmarks.py tests/test_vtm_rlm.py tests/test_verification.py tests/test_retrieval.py tests/test_types.py`
- `uv run --extra dev ruff check src tests`
- `uv run python -m compileall -q src tests`

## Known limitations

- SWE-bench Lite was attempted as an external agent pilot, but it is not part of the maintained result surface or paper claim.
- DSPy is integrated as an optional agent-facing layer, but the quantitative evidence still comes from retrieval, drift verification, drifted retrieval, and controlled coding-drift.
- LiveCodeBench is useful for baseline model capability checks, but it is not a direct VTM memory benchmark.

## Future work

- SWE-bench can be revisited later as an external benchmark, but it is not part of the final paper result.
- LongCoT can be revisited later as a reasoning-side pilot, but it is not maintained in this final benchmark surface.
- DSPy can sit on top of VTM as an optional agent scaffold without changing the frozen benchmark evidence.
