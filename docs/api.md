# API Notes

## Store protocols

- `MetadataStore`
  - save/get/list/query memory items
  - save/list lineage edges
  - save/get/list transactions
  - append/list/move/clear staged memory items for active transactions
  - run grouped SQLite mutations through `run_atomically(...)`
- `EventStore`
  - save/get/list memory events
- `ArtifactStore`
  - prepare bytes into a recoverable artifact capture record
  - commit or abandon a prepared capture
  - write bytes as capture records over SHA-256-addressed blobs
  - fetch the latest artifact record by SHA-256
  - list all artifact records for a SHA-256 digest
  - list artifact records by capture state
  - audit prepared captures, committed missing blobs, and orphaned blob paths without mutating the store
  - read bytes by artifact id
  - abandon stale prepared captures and clean up orphaned blobs
- `CacheStore`
  - save/get/delete/list cache entries
- `EmbeddingIndexStore`
  - save/get/list/delete derived embedding rows keyed by `(memory_id, adapter_id)`

## Services

- `TransactionalMemoryKernel`
  - `begin_transaction`
  - `build_code_anchor`
  - `anchor_evidence`
  - `capture_artifact`
  - `artifact_evidence`
  - `stage_memory_item`
  - `list_visible_memory`
  - `commit_transaction`
  - `rollback_transaction`
  - `verify_memory`
  - `validate_procedure`
  - `promote_to_procedure`
  - `retrieve`
  - `expand`
  - `save_cache_entry` / `get_cache_entry`
- `DependencyFingerprintBuilder`
  - combines live repo and env collectors with caller-supplied dependency IDs and input digests
- `BasicVerifier`
  - verifies memory items against a current dependency fingerprint
  - supports parser-agnostic relocation with Python Tree-sitter as the primary path and Python AST fallback
- `LexicalRetriever`
  - queries persisted memory items by scope and status
  - ranks deterministically from lexical overlap
  - exposes `expand(memory_id)` to fetch raw evidence
- `EmbeddingRetriever`
  - queries the same committed-memory set as lexical retrieval
  - lazily backfills and refreshes derived embedding vectors through `EmbeddingIndexStore`
  - ranks by cosine similarity, then lexical overlap, then `updated_at DESC`, then `memory_id ASC`
  - exposes `expand(memory_id)` to fetch raw evidence
- `RLMRerankingRetriever`
  - wraps any `Retriever`
  - fans lexical retrieval out to a configurable top-k
  - reranks the lexical candidate set through `RLMAdapter.rank_candidates(...)`
  - preserves lexical ordering if reranking fails
  - can cache rerank responses through `CacheStore` using query, candidate digest, and repo/env fingerprints
- `CommandProcedureValidator`
  - executes `ValidatorSpec(kind="command")` procedures locally
  - captures stdout and stderr as artifacts
  - supports `timeout_seconds`, `max_output_bytes`, `env_allowlist`, and `env_denylist`
  - returns `ProcedureValidationResult`; kernel writeback persists the latest validation summary into memory metadata
- `DeterministicConsolidator`
  - scans committed memory outside transaction commit paths
  - supersedes duplicate verified or relocated memories within the same visibility scope
  - emits consolidation lineage edges and events
  - can generate deterministic `summary_card` memories over consolidation groups

Active transactions persist both their transaction records and staged memory items in SQLite. Constructing a fresh kernel over the same stores preserves staged visibility until the transaction is committed or rolled back.

`TransactionalMemoryKernel(...)` defaults to `require_shared_event_store=True`. When enabled, the kernel requires `event_store is metadata_store` and that both are `SqliteMetadataStore`, making transaction state changes, verification writeback, procedure validation metadata updates, procedure promotion, retrieval-stat updates, and corresponding SQLite event rows atomic.

## Concrete implementations

- `GitRepoFingerprintCollector()`
- `PythonTreeSitterSyntaxAdapter(fallback=PythonAstSyntaxAdapter())`
- `PythonAstSyntaxAdapter()`
- `PythonAstAnchorAdapter()` / `PythonAstAnchorRelocator()` remain compatibility wrappers
- `RuntimeEnvFingerprintCollector()`
- `DeterministicHashEmbeddingAdapter(dimensions=64)`
- `OpenAIEmbeddingAdapter(model=...)`
  - optional extra dependency: `openai`
  - uses the OpenAI embeddings API
  - requires `api_key` or `OPENAI_API_KEY`
- `OpenAIRLMAdapter(model=...)`
  - optional extra dependency: `openai`
  - uses the OpenAI Responses API through structured JSON output
  - requires `api_key` or `OPENAI_API_KEY`
- `CommandProcedureValidator(artifact_store)`
- `SqliteMetadataStore(db_path, event_log_path=...)`
  - SQLite is the canonical event ledger
  - `export_events_to_jsonl()` is an at-least-once append export; retries can duplicate lines by `event_id`
  - `rebuild_events_jsonl()` rewrites a deduped JSONL log from SQLite source-of-truth rows
- `FilesystemArtifactStore(root)`
  - repeated identical bytes create distinct capture records while reusing the same blob path
  - `audit_integrity()` reports prepared captures, committed missing blobs, and orphaned blob paths
  - prepared captures can be committed, abandoned, and cleaned up explicitly
- `SqliteCacheStore(db_path, event_store=...)`
- `SqliteEmbeddingIndexStore(db_path)`

All persisted public shapes support `to_json()` and `from_json()` for round-trip tests and storage boundaries.

## Retrieval and consolidation records

- `EmbeddingIndexEntry`
  - `memory_id`
  - `adapter_id`
  - `content_digest`
  - `vector`
  - `created_at` / `updated_at`
- `ConsolidationAction`
  - `action_type`
  - `canonical_memory_id`
  - `affected_memory_ids`
  - optional `created_memory_id`
  - `metadata`
- `ConsolidationRunResult`
  - scanned memory count, candidate group count, action count
  - emitted actions
  - start and completion timestamps

## RLM contracts

- `RLMRankRequest`
  - `query`
  - `candidates`
  - `top_k`
  - `metadata`
- `RLMRankedCandidate`
  - candidate identifier plus narrow auditable retrieval fields
  - lexical score, optional RLM score, optional final score, and optional reason
- `RLMRankResponse`
  - ranked candidates
  - optional model name, usage, and provider metadata
- `RLMAdapter`
  - `rank_candidates(request) -> RLMRankResponse`

## Benchmark package

- `vtm.benchmarks.models`
  - `RepoSpec`, `CommitPair`, `RetrievalCase`, `DriftCase`, `CodingTaskCase`
  - `BenchmarkManifest`, `BenchmarkRunConfig`, `BenchmarkCaseResult`, `BenchmarkRunResult`
  - `RetrievalCase.slice_name` distinguishes `taskish_behavior` from `smoke_identity` retrieval slices
  - `BenchmarkRunConfig.repo_filters` / `pair_filters` scope a run before case generation
  - `CodingTaskCase.expected_changed_paths` defaults to `touched_paths` when omitted
  - `CodingTaskCase.task_kind` and `difficulty` carry descriptive task metadata for coding benchmarks
  - `CodingTaskCase.evaluation_backend` distinguishes local subprocess scoring from official SWE-bench harness scoring
  - `CodingTaskCase.instance_id`, `dataset_name`, `problem_statement`, `hints_text`, `fail_to_pass_tests`, `pass_to_pass_tests`, and `gold_test_patch_digest` capture SWE-bench-specific coding metadata
  - `BenchmarkRunConfig.swebench_dataset_name`, `swebench_harness_workers`, `swebench_harness_cache_level`, and `swebench_harness_run_id` configure harness-backed coding runs
- `vtm.benchmarks.runner.BenchmarkRunner`
  - remains the public benchmark entrypoint
  - delegates internally to repo materialization, symbol indexing, suite execution, reporting, and subprocess executor helpers
  - materializes repo sources and generates retrieval/drift cases deterministically
  - validates that persisted benchmark cases and results stay aligned one-to-one
  - supports retrieval `mode` values `no_memory`, `lexical`, `embedding`, and `lexical_rlm_rerank`
  - writes `manifest.lock.json`, `cases.jsonl`, `results.jsonl`, `summary.json`, and `summary.md`
- CLI
  - `python -m vtm.benchmarks.run --manifest ... --suite ... --output ... --top-k ... --max-cases ...`
  - optional flags: `--mode`, `--seed`, `--repo`, `--pair`, `--rlm-model`, `--embedding-model`, `--executor-command`, `--swebench-dataset-name`, `--swebench-harness-workers`, `--swebench-cache-level`, `--swebench-run-id`
  - `python -m vtm.benchmarks.prepare_swebench_lite --output-manifest ... --cache-root ...`
- Subprocess executor wrapper
  - keeps `--executor-command` backward compatible
  - writes benchmark-local executor stdout, stderr, test stdout, test stderr, and produced patch files under the benchmark output directory
  - records produced changed paths from the workspace diff so coding-task scoring can compare actual edits against expected paths
  - the checked-in `scripts/vtm_local_patcher.py` script is a single-shot OpenAI-compatible patch generator intended for local vLLM or LM Studio serving

## Coding benchmark task packs and metrics

- Task pack JSON
  - `case_id`, `repo_name`, `commit_pair_id`
  - `evaluation_backend`, `instance_id`, `dataset_name`
  - `base_ref`, `head_ref`, optional `commit_pair_label`
  - `task_statement`, `problem_statement`, `hints_text`
  - `failing_tests`, `fail_to_pass_tests`, `pass_to_pass_tests`, `test_command`
  - `expected_changed_paths`, `touched_paths`
  - `target_patch_digest`, optional `gold_test_patch_digest`
  - `memory_mode`, `top_k`
  - `task_kind`, `difficulty`
  - `memory_context`
- Coding result metrics
  - `passed` is the primary benchmark outcome for testable tasks
  - `resolved` tracks official SWE-bench harness resolution for harness-backed tasks
  - `executor_succeeded` indicates whether the executor command itself exited successfully
  - `produced_patch_nonempty` indicates whether the workspace diff after execution was non-empty
  - `patch_applied` tracks whether the harness or local subprocess path observed an applied patch
  - `changed_path_precision`, `changed_path_recall`, and `changed_path_f1` compare actual changed paths against `expected_changed_paths`
  - `patch_similarity`, `runtime_ms`, `retrieval_usage_rate`, and `context_chars` remain secondary diagnostics
  - harness-backed runs additionally write `predictions.jsonl`, `swebench_harness_results.json`, and a harness logs directory into the benchmark output root

## Recovery workflow

- For event export, treat SQLite as canonical, JSONL as at-least-once, dedupe by `event_id`, and use `rebuild_events_jsonl()` as the repair path.
- For artifact/storage divergence, call `audit_integrity()` first, then use `abandon_stale_prepared_artifacts()` and `cleanup_orphaned_blobs()` as explicit repair steps.
