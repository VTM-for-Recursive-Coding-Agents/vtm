# docs/decisions

Purpose: architecture decision records for behaviors that should stay stable across refactors.

Contents
- `0001-kernel-first.md`: Explains the kernel-first approach and why correctness boundaries came before learned retrieval or optimization work.
- `0002-event-and-artifact-contracts.md`: Defines the canonical SQLite event ledger, JSONL export semantics, and artifact capture lifecycle guarantees.
- `0003-benchmark-harness-contract.md`: Defines the benchmark package boundary and the persisted benchmark output contract.
- `0004-rlm-reranking-contract.md`: Documents the provider-neutral reranking interface and why model reranking stays outside the core retrieval API.
- `0005-schema-compatibility-policy.md`: Defines forward-only SQLite schema upgrades, required migration fixtures, and future-version rejection policy.
- `0006-embedding-index-contract.md`: Documents derived embedding indexing, benchmark embedding mode, and the auditable embedding text surface.
- `0007-deterministic-consolidation.md`: Documents duplicate superseding, summary-card generation, and intentionally conservative consolidation boundaries.
