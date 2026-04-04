# ADR 0006: Embedding Index Contract

## Status

Accepted

## Context

VTM now needs an embedding-backed retrieval mode, but the kernel should not blur the line between canonical memory records and derived retrieval indexes. Embedding generation is adapter-specific, can be refreshed lazily, and should stay auditable.

## Decision

- Keep canonical memory records in `SqliteMetadataStore`.
- Add a separate `SqliteEmbeddingIndexStore` keyed by `(memory_id, adapter_id)`.
- Treat embedding rows as derived state that can be rebuilt from committed memory.
- Restrict embedding source text to auditable fields only: title, summary, tags, validity status, and optional code-anchor path, symbol, kind, and language.
- Add `EmbeddingRetriever` without changing `TransactionalMemoryKernel.retrieve(...)`.
- Support a deterministic local embedding adapter plus an optional OpenAI adapter behind the existing `openai` extra.

## Consequences

- Embedding rows can be rebuilt lazily and refreshed when the derived content digest changes.
- Retrieval can compare lexical and embedding modes through the existing benchmark harness.
- Embedding retrieval remains explicitly derived and does not change the canonical transaction or event contracts.
