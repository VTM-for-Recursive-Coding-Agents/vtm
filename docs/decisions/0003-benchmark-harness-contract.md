# ADR 0003: Benchmark Harness Contract

## Status

Accepted

## Context

VTM now needs reproducible evaluation on real repositories and deterministic synthetic fixtures without polluting production metadata, events, or artifact stores. Benchmark runs also need a stable on-disk contract so results can be compared across retrieval modes and commit pairs.

## Decision

- Add a dedicated benchmark package under `vtm.benchmarks`.
- Keep benchmark outputs outside the production VTM store layout.
- Allow benchmark runs to filter manifests down to selected repos and commit pairs without editing the checked-in manifest.
- Keep retrieval cases explicit about their slice so task-oriented and smoke identity queries can coexist in the same persisted run.
- Persist benchmark runs as:
  - `manifest.lock.json`
  - `cases.jsonl`
  - `results.jsonl`
  - `summary.json`
  - `summary.md`
- Support two corpus sources:
  - pinned Git repositories
  - a deterministic synthetic Python smoke corpus generated locally

## Consequences

- Benchmark runs are reproducible from pinned manifests and synthetic fixtures.
- `manifest.lock.json` captures filter inputs when callers scope a run to selected repos or commit pairs.
- `case_count` in persisted summaries tracks the selected persisted cases rather than any intermediate result-row count.
- Benchmark artifacts do not mix with canonical metadata/event storage.
- OSS benchmark runs may require network access and are better suited to manual or scheduled execution than every PR CI run.
