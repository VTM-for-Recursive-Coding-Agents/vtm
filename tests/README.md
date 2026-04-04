# tests

Purpose: regression coverage for the public record layer, the kernel/service behavior, storage semantics, benchmark harness, adapter behavior, and documentation parity.

Contents
- `conftest.py`: Shared pytest fixtures for stores, kernels, fingerprints, evidence, embedding retrieval, and memory factories.
- `test_anchor_relocation.py`: Verifier-focused relocation tests for whitespace-only anchor moves.
- `test_artifacts.py`: Artifact blob reuse, capture lifecycle, integrity audit, and cleanup coverage.
- `test_benchmark_cli.py`: Smoke tests for the benchmark CLI entrypoint, including deterministic embedding mode.
- `test_benchmarks.py`: Benchmark runner integration tests, deterministic run-id coverage, duplicate-symbol benchmark case-ID coverage, retrieval-slice coverage, reranking benchmark coverage, expanded synthetic coding-task coverage, filter validation, embedding mode coverage, changed-path scoring checks, and executor artifact capture checks.
- `test_cache.py`: Cache key normalization, expiry, freshness, and cache hit/miss logging tests.
- `test_consolidation.py`: Deterministic consolidation, idempotency, lineage, summary-card, and superseded-retrieval coverage.
- `test_docs_parity.py`: Executable docs example, repo-wide markdown link validation, and manifest reference checks.
- `test_embeddings.py`: Embedding retrieval ranking, lazy index refresh, and status-filter coverage.
- `test_event_export.py`: JSONL event export ordering, at-least-once semantics, and rebuild-path coverage.
- `test_kernel_topology.py`: Shared-event-store requirements and degraded-topology opt-in behavior.
- `test_memory_kernel_runtime.py`: End-to-end runtime flow covering artifact capture, anchoring, retrieval, and re-verification.
- `test_openai_embedding.py`: OpenAI embedding adapter request and response-shape tests with a fake client.
- `test_openai_rlm.py`: OpenAI reranking adapter request and response-shape tests with a fake client.
- `test_procedures.py`: Command validator success, failure, timeout, truncation, and environment-handling coverage.
- `test_python_anchors.py`: Python AST and Tree-sitter anchor build and relocation parity tests.
- `test_retrieval.py`: Retrieval filtering, evidence-budget behavior, and lexical ranking coverage.
- `test_rlm_reranking.py`: Reranking wrapper ordering, caching, failure fallback, and expand passthrough tests.
- `test_runtime_fingerprints.py`: Git dirty-state and runtime fingerprint collection tests.
- `test_service_atomicity.py`: Atomic rollback guarantees when service-layer event persistence fails.
- `test_store_migrations.py`: Fixture-backed migration upgrades and future-schema rejection tests for metadata, cache, artifact, and embedding stores.
- `test_swebench.py`: SWE-bench Lite manifest preparation, local patcher, and fake harness-backed coding benchmark coverage.
- `test_transactions.py`: Nested transaction persistence, commit, rollback, and restart behavior.
- `test_types.py`: Model round-trip, validator, and package import smoke coverage.
- `test_verification.py`: Verification status transitions for unchanged, changed, stale, and relocated dependencies.
- `test_visibility.py`: Transaction visibility isolation between parents and siblings.
- `fixtures/`: Checked-in test assets, primarily schema migration fixtures.
