# Final Scope

VTM is now scoped as a clean study of verified lexical repository memory for coding agents, with OpenRouter as the only maintained inference path.

## Main claim

Verified lexical memory should outperform a no-memory baseline without relying on oracle localization hints in model-visible inputs.

## Kept benchmark modes

- `no_memory`
- `naive_lexical`
- `verified_lexical`
- optional `lexical_rlm_rerank` as a thin secondary ablation

## Removed from scope

- Embedding retrieval
- Terminal-only benchmark tracks
- Broad provider experimentation and local ad hoc model routing
- Codex execution paths
- Large generated documentation and compatibility shims that only preserved old surfaces

## Why the final evaluation layers remain

- Retrieval measures whether the right repository memory can be found.
- Drift measures whether stored memory stays valid under repository change.
- Drifted retrieval measures whether useful repository memory can still be found after repository change.

These three layers isolate memory quality and memory freshness cleanly enough for the paper.

## External pilot status

- SWE-bench Lite is no longer a main paper result.
- The maintained SWE-bench Lite code stays in the repository as optional harness infrastructure and local reference material.
- LongCoT-Mini CS may be used as a small optional long-horizon reasoning pilot because it has deterministic answer verification and does not require a patch-generation harness.
- Main VTM evidence remains retrieval, drift, and drifted retrieval.

Synthetic smoke tasks remain only as a maintained local/dev validation path for the OpenRouter-backed executor and table-export workflow.

## OpenRouter defaults

- Base URL env var: `VTM_OPENROUTER_BASE_URL`
- Default base URL: `https://openrouter.ai/api/v1`
- API key env var: `OPENROUTER_API_KEY`
- Execution model env var: `VTM_EXECUTION_MODEL`
- Rerank model env var: `VTM_RERANK_MODEL`
- Smoke/dev rerank model: `nvidia/nemotron-3-nano-30b-a3b:free`
- Default execution model: `google/gemma-4-31b-it:free`
- Optional stronger ablation model: `nvidia/nemotron-3-super-120b-a12b:free`
