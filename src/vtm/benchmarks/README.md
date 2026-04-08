# src/vtm/benchmarks

Purpose: manifest-driven benchmark orchestration and reporting.

Start here
- `models.py`: manifest, config, and result records.
- `runner.py`: the public `BenchmarkRunner` entrypoint.
- `suite_execution.py`: how retrieval, drift, and coding suites are dispatched.

Contents
- `models.py`: Manifest, case, config, and result records.
- `runner.py`: Public `BenchmarkRunner`.
- `suite_execution.py`: High-level suite dispatcher.
- `retrieval_suite.py`, `drift_suite.py`, `coding_suite.py`: Suite-specific execution logic.
- `kernel_factory.py`: Benchmark-local kernel setup and seeding helpers.
- `reporting.py`: Aggregate metrics and summary rendering.
- `repo_materialization.py`, `symbol_index.py`, `synthetic.py`, `swebench.py`, `swebench_harness.py`: corpus preparation and evaluation helpers.
- `executor.py`: Compatibility shim re-exporting the public harness executor surface.
- `local_patcher.py`: Local patch generator that consumes typed harness task packs.

Current benchmark credibility surface
- `benchmarks/manifests/synthetic-smoke.json`: small regression-friendly smoke corpus.
- `benchmarks/manifests/terminal-smoke.json`: harder local terminal-style coding corpus.
- `BenchmarkRunConfig.attempt_count` and `pass_k_values`: repeated-attempt coding controls.
- `results.jsonl`: one aggregate row per case.
- `attempts.jsonl`: one row per attempt for coding suites.
