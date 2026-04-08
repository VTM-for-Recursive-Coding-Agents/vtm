# ADR 0010: Multi-Attempt Coding Benchmarks

## Status

Accepted

## Context

Single-run coding results were enough for smoke validation, but they were not a credible way to compare native-agent prompting, memory modes, or RLM reranking against harder terminal-style tasks. The benchmark runner needed repeated attempts, stable per-attempt artifacts, and explicit `pass@k` style reporting without changing the existing one-row-per-case aggregate contract.

## Decision

- Keep repeated attempts scoped to coding suites.
- Keep `task-packs/<case-id>.json` canonical per case rather than per attempt.
- Materialize one isolated workspace and one artifact root per attempt:
  - `workspaces/<mode>/<case-id>/attempt-01`
  - `executor-artifacts/<case-id>/attempt-01`
- Write:
  - `results.jsonl` with one aggregate row per case
  - `attempts.jsonl` with one row per attempt
- Treat attempt `1` as the canonical debugging run while still aggregating `pass_at_k`, `resolved_at_k`, and `patch_applied_at_k`.
- Keep repeated-attempt semantics local and harness-native for this phase; remote sandboxing and multi-agent orchestration remain out of scope.

## Consequences

- Benchmark runs now support credible repeated-attempt evaluation for the local coding harness.
- External executors and native-agent runs share the same attempt-local workspace and artifact contract.
- Benchmark recipes can compare `no_memory`, `lexical`, `lexical_rlm_rerank`, and `embedding` on the same harder terminal-style track.
- The feature remains intentionally narrow: retrieval and drift suites stay single-attempt, and shell-only task classes are still future work.
