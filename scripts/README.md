# scripts

Purpose: reserved location for developer and maintenance automation that should live with the repository.

Maintained optional paper-adjacent scripts:

- `run_longcot_pilot.py`: isolated external LongCoT pilot runner; not part of the main VTM benchmark matrix
- `run_controlled_coding_drift.sh`: maintained strong-signal VTM regression wrapper for the controlled coding-drift matrix plus report export
- `run_maintained_regression_bundle.sh`: maintained regression bundle wrapper for retrieval, drifted retrieval, drift, controlled coding-drift, and paper-table exports
- `run_livecodebench_baseline.sh`: external LiveCodeBench baseline wrapper using the repo's OpenRouter conventions
- `run_livecodebench_dspy_pilot.py`: small LiveCodeBench pilot comparing direct, DSPy ReAct and DSPy RLM baselines, and the corresponding VTM-memory variants
- `run_livecodebench_dspy_pilot_batch.sh`: helper wrapper for fixed-size LiveCodeBench DSPy pilot batches; defaults to 25 problems per batch
- `run_dspy_vtm_smoke.py`: dry-run smoke script for the optional DSPy plus VTM integration
- `livecodebench/export_dspy_pilot_results.py`: export pilot summaries into `.benchmarks/paper-tables/livecodebench-dspy-pilot`
- `livecodebench/setup_livecodebench.sh`: clone and install the external LiveCodeBench checkout under `benchmarks/LiveCodeBench`
  The checkout is local external state and should remain untracked.
