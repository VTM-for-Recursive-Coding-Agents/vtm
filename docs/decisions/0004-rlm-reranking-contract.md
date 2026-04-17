# ADR 0004: RLM Reranking Contract

## Status

Accepted

## Context

The next capability step is model-assisted retrieval, but the kernel should not become provider-specific or change its public retrieval surface just to accommodate one RLM. The benchmark harness also needs a stable way to compare lexical retrieval against model-assisted reranking.

## Decision

- Introduce a provider-neutral `RLMAdapter` contract centered on reranking.
- Keep `TransactionalMemoryKernel.retrieve(...)` unchanged.
- Add `RLMRerankingRetriever` as a wrapper over the existing `Retriever` protocol.
- Restrict RLM inputs to narrow auditable fields:
  - query
  - title
  - summary
  - tags
  - validity status
  - optional anchor path and symbol
- Cache rerank responses with query, candidate digest, and repo/env fingerprints.
- Provide OpenAI as an optional reference adapter, not a mandatory dependency.

## Consequences

- Lexical retrieval remains the default and the benchmark baseline.
- Provider-specific logic stays at the adapter boundary.
- Additional retrieval backends remain out of scope rather than leaking into the reranking contract prematurely.
