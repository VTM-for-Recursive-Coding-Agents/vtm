# ADR 0001: Kernel First

## Status

Accepted

## Context

VTM is intended for coding agents that work inside mutable repositories. The first implementation pass needs to establish correct state boundaries before any learned retrieval, benchmark integration, or performance tuning can matter.

## Decision

Build the system as a kernel-first scaffold:

- frozen Pydantic models for all persisted records
- explicit store and service interfaces
- SQLite for metadata, transactions, events, and cache
- filesystem content-addressed artifact storage
- SQLite as the canonical event ledger with optional JSONL export
- minimal but deterministic verification and retrieval behavior

## Consequences

- The repository accepts a hard break from the initial scratch `vtm/types.py` experiment.
- Git fingerprinting landed as part of the kernel baseline.
- Python Tree-sitter anchors landed once the event/export and artifact lifecycle contracts were explicit.
- The maintained scope stays correctness-first: lexical retrieval, deterministic verification, and durable kernel contracts came before any broader provider or retrieval expansion.
- Tests emphasize round-trip correctness, transaction semantics, retrieval defaults, and verification transitions instead of benchmark outcomes.
