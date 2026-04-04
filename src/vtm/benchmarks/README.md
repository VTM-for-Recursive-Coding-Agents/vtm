# src/vtm/benchmarks

Purpose: benchmark harness for measuring retrieval quality, verification drift behavior, and optional coding-task workflows against pinned repositories.

Current retrieval evaluation writes both `taskish_behavior` and `smoke_identity` slices so the same run can separate harder prompt-style lookup from exact identity smoke coverage.

Current coding evaluation supports both local deterministic synthetic tasks and harness-backed SWE-bench Lite tasks. Task packs now carry base/head refs, expected changed paths, target patch digests, memory mode metadata, richer retrieval context, and optional SWE-bench dataset metadata, while coding summaries prioritize pass rate and resolved-rate style comparisons.

Contents
- `__init__.py`: Re-exports the public benchmark models and `BenchmarkRunner`.
- `executor.py`: Internal executor protocol plus the subprocess-backed implementation used for `--executor-command`.
- `models.py`: Manifest, repo, commit-pair, case, config, and run-result records for the harness.
- `local_patcher.py`: Reusable single-shot OpenAI-compatible patcher used by the checked-in script wrapper.
- `repo_materialization.py`: Clones, fetches, checks out, and diffs benchmark repositories or synthetic corpora.
- `reporting.py`: Aggregates suite metrics and writes human-readable and JSONL benchmark summaries.
- `prepare_swebench_lite.py`: CLI entrypoint for generating SWE-bench Lite manifests backed by local repo caches.
- `run.py`: CLI entrypoint that parses arguments and launches a benchmark run, including optional repo/pair filters plus embedding and RLM adapter wiring.
- `runner.py`: High-level benchmark orchestration that writes lockfiles, cases, results, and summaries.
- `suite_execution.py`: Core retrieval, drift, and coding-task execution logic, including benchmark-local kernel setup, case filtering, coding task-pack generation, changed-path scoring, embedding mode, and executor artifact capture.
- `swebench.py`: SWE-bench Lite dataset normalization, repo-cache preparation, synthetic gold-ref creation, and manifest generation helpers.
- `swebench_harness.py`: Official SWE-bench harness prediction writing, invocation, and result normalization.
- `symbol_index.py`: Extracts Python symbols and derives deterministic retrieval and drift cases from them, including harder taskish queries and smoke identity queries.
- `synthetic.py`: Builds the synthetic Python smoke corpus used by tests and local smoke runs, including a multi-task coding benchmark suite.
