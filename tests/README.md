# tests

Purpose: regression coverage for kernel behavior, harness contracts, vendored-RLM integration, and benchmark workflows.

Contents
- `test_types.py`: Model round-trip, validator, and package import smoke tests.
- `test_harness.py`: Harness contract round-trip and workspace-driver behavior.
- `test_benchmarks.py`: Benchmark runner integration, coding-task execution, and artifact layout.
- `test_vtm_rlm.py`: Vendored-RLM bridge, executor smoke, and memory writeback coverage.
- Remaining `test_*.py` files: storage, migrations, retrieval, transactions, verification, consolidation, adapters, and SWE-bench coverage.
