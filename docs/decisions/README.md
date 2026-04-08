# docs/decisions

Purpose: architecture decision records for stable boundaries, compatibility policy, and long-lived contracts.

Contents
- `0001-kernel-first.md`: Kernel-first foundation and why correctness boundaries came first.
- `0002-event-and-artifact-contracts.md`: Canonical event ledger and artifact lifecycle guarantees.
- `0003-benchmark-harness-contract.md`: Benchmark output boundary and persisted run contract.
- `0004-rlm-reranking-contract.md`: Provider-neutral reranking interface.
- `0005-schema-compatibility-policy.md`: Forward-only schema upgrades and future-version rejection.
- `0006-embedding-index-contract.md`: Derived embedding index contract.
- `0007-deterministic-consolidation.md`: Deterministic duplicate superseding and summary-card policy.
- `0008-agent-runtime-boundary.md`: Native agent runtime boundary and task-memory promotion rules.
- `0009-harness-executor-boundary.md`: Public harness seam for task packs, workspaces, executors, and traces.
- `0010-multi-attempt-coding-benchmarks.md`: Repeated-attempt coding benchmark semantics and artifact layout.
