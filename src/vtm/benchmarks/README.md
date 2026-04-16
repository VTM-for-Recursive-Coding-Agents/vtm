# src/vtm/benchmarks

Purpose: manifest-driven benchmark orchestration and reporting.

Start here
- `models.py`: manifest, config, and result records.
- `runner.py`: the public `BenchmarkRunner` entrypoint.
- `suite_execution.py`: how retrieval, drift, and coding suites are dispatched.
- `run.py`: CLI entrypoint exposed as `vtm-bench`.
- `matrix.py`: maintained mode-matrix CLI exposed as `vtm-bench-matrix`.

Contents
- `models.py`: Manifest, case, config, and result records.
- `runner.py`: Public `BenchmarkRunner`.
- `suite_execution.py`: High-level suite dispatcher.
- `retrieval_suite.py`, `drift_suite.py`, `coding_suite.py`: Suite-specific execution logic.
- `kernel_factory.py`: Benchmark-local kernel setup and seeding helpers.
- `reporting.py`: Aggregate metrics, paired comparisons, and summary rendering.
- `compare.py`: CLI entrypoint for offline comparison of completed benchmark runs.
- `matrix.py`: CLI entrypoint for maintained benchmark matrices and baseline comparisons.
- `repo_materialization.py`, `symbol_index.py`, `synthetic.py`, `swebench.py`, `swebench_harness.py`: corpus preparation and evaluation helpers.
- `executor.py`: Compatibility shim re-exporting the public harness executor surface.
- `local_patcher.py`: Local patch generator that consumes typed harness task packs.

Current benchmark credibility surface
- `benchmarks/manifests/synthetic-smoke.json`: small regression-friendly smoke corpus.
- `benchmarks/manifests/terminal-smoke.json`: harder local terminal-style coding corpus.
- `benchmarks/manifests/terminal-shell-smoke.json`: shell-command coding corpus intended to be solved from the terminal.
- `BenchmarkRunConfig.attempt_count` and `pass_k_values`: repeated-attempt coding controls.
- `BenchmarkRunConfig.workspace_backend`: `local_workspace` or `docker_workspace`.
- `results.jsonl`: one aggregate row per case.
- `attempts.jsonl`: one row per attempt for coding suites.
- `vtm-bench-compare`: offline comparison CLI that emits `comparison.json` and `comparison.md`.
- `vtm-bench-matrix`: preset-driven matrix CLI that emits `matrix.json` and `matrix.md`.
- coding summaries now break down results by `execution_style` and `workspace_backend`.
