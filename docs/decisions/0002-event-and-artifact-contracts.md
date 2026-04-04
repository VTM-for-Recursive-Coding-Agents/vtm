# ADR 0002: Event And Artifact Contracts

## Status

Accepted

## Context

The kernel now has real persistence and verification behavior, so the remaining ambiguity is no longer about whether events and artifacts exist, but about what guarantees they actually provide. JSONL export cannot be made atomic with SQLite commits, and filesystem artifact writes cannot share a single atomic boundary with SQLite metadata.

## Decision

- Keep SQLite as the canonical event ledger.
- Treat JSONL as an at-least-once derived export; consumers dedupe by `event_id` and `rebuild_events_jsonl()` is the repair path.
- Require `event_store is metadata_store` by default so the strongest metadata/event atomicity is the standard kernel topology.
- Represent artifact captures explicitly as `prepared`, `committed`, or `abandoned` so cross-store failures leave recoverable state rather than silent ambiguity.
- Add a non-mutating artifact integrity audit so operators can inspect prepared captures, committed missing blobs, and orphaned blobs before cleanup.

## Consequences

- JSONL consumers must tolerate duplicates by `event_id`; rebuild from SQLite is the canonical repair workflow.
- Cross-store atomicity is still not available, but capture recovery is inspectable and testable through explicit state plus artifact integrity audits.
- Degraded event-store topologies still exist, but callers must opt into them deliberately.
