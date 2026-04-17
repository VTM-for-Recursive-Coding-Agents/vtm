# Final Scope

VTM is now scoped as a fair, testable study of repository memory for coding agents.

## Main claim

The primary experimental question is whether verified repository memory improves coding outcomes over no memory and over naive lexical memory, without leaking oracle localization hints into model-visible inputs.

## Benchmark modes

- `no_memory`: no retrieved memory is shown to the agent.
- `naive_lexical`: lexical retrieval can surface visible committed memory without retrieval-time validity gating or verify-on-read.
- `verified_lexical`: lexical retrieval verifies or relocates candidate memories against the current repository state before surfacing them and only returns `verified` / `relocated` memories.
- `lexical_rlm_rerank`: optional secondary ablation that reranks the verified lexical candidate set without expanding the study scope beyond memory as the experimental variable.

`lexical` remains a compatibility alias for `verified_lexical`.

## Removed from scope

- Oracle changed-path hints in external coding prompts.
- Broad RLM productization beyond a thin execution bridge or reranker.
- Dynamic memory-tool injection for Codex runs.
- Treating embedding retrieval as part of the maintained headline study.

## Evaluation layers

- Retrieval: does the memory system return the right repository memory items?
- Drift: do stored memories stay valid, relocate correctly, or get filtered as stale?
- Coding: does memory help an agent solve benchmark tasks under the same visible task information?

These three layers are the final evaluation stack because they isolate memory quality, memory freshness, and end-task utility without conflating them.
