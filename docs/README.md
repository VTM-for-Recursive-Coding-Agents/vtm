# docs

Purpose: durable architecture, contract, and maintainer-facing reference documents for the VTM kernel.

Contents
- `api.md`: Public API notes for store protocols, service entrypoints, benchmark entrypoints, and concrete implementations.
- `architecture.md`: Layered architecture overview plus transaction, verification, retrieval, consolidation, and benchmark flows.
- `benchmark-recipes.md`: Checked-in benchmark commands for tonight baselines, full synthetic sweeps, targeted OSS runs, and SWE-bench Lite prep/run workflows.
- `benchmark-results-template.md`: Maintainer template for recording benchmark runs and comparing results across modes.
- `current-state-audit.md`: Maintainer status snapshot of implemented guarantees, remaining gaps, and intentionally limited areas.
- `runtime-example.md`: End-to-end executable example showing a realistic kernel setup and memory flow.
- `swebench-lite-windows.md`: Windows and WSL2 runbook for preparing and running the SWE-bench Lite workflow on a machine with a local model server.
- `type-system.md`: Record and enum reference with the core invariants enforced by Pydantic validators.
- `decisions/`: ADRs that capture long-lived design decisions, compatibility policy, embedding indexing, and consolidation semantics.
