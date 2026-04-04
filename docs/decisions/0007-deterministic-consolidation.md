# ADR 0007: Deterministic Consolidation

## Status

Accepted

## Context

VTM needs a first implemented consolidation policy, but the initial step should be conservative, auditable, and fully deterministic. Learned summarization, TTL forgetting, and deletion policies are broader product decisions and should not be smuggled in as maintenance behavior.

## Decision

- Add `DeterministicConsolidator` as an explicit maintenance service, not a transaction-commit hook.
- Consolidate only duplicate verified or relocated memories within the same visibility scope and memory kind.
- Group duplicates by normalized title, normalized summary, normalized tags, kind, visibility, and dependency fingerprint digest.
- Keep the newest active memory as canonical and mark older active duplicates as `superseded`.
- Persist lineage edges and consolidation events for every superseded memory.
- Optionally generate deterministic `summary_card` memories that reference supporting memories through memory evidence refs.

## Consequences

- Consolidation is implemented and testable without introducing nondeterministic model behavior.
- Retrieval defaults continue to hide superseded memories because they are outside the default retrieval status set.
- Advanced forgetting, learned summarization, deletion, and cross-scope merging remain separate future policy work.
