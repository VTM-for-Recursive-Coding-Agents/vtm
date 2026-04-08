# tests

Purpose: regression coverage for kernel behavior, harness contracts, native-agent runtime behavior, benchmark workflows, and docs parity.

Contents
- `test_types.py`: Model round-trip, validator, and package import smoke tests.
- `test_harness.py`: Harness contract round-trip, shim compatibility, and workspace-driver behavior.
- `test_agents.py`: Native-agent runtime, tools, and permission policies.
- `test_benchmarks.py`: Benchmark runner integration, coding-task execution, and artifact layout.
- `test_docs_parity.py`: Runtime example execution plus markdown, manifest, and boundary-doc checks.
- Remaining `test_*.py` files: storage, migrations, retrieval, transactions, verification, consolidation, adapters, and SWE-bench coverage.
