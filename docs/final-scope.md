# Final Scope

VTM is now scoped as a clean study of verified lexical repository memory for coding agents, with OpenRouter as the only maintained inference path.

## Main claim

Verified lexical memory should outperform a no-memory baseline without relying on oracle localization hints in model-visible inputs.

## Kept benchmark modes

- `no_memory`
- `naive_lexical`
- `verified_lexical`
- optional `lexical_rlm_rerank` as a thin secondary ablation

## Final maintained benchmark surface

1. static retrieval
   `no_memory`, `naive_lexical`, `verified_lexical`
2. drift verification
   `verified_lexical`
3. drifted retrieval
   `no_memory`, `naive_lexical`, `verified_lexical`
4. controlled coding-drift
   `no_memory`, `naive_lexical`, `verified_lexical`

## Removed from scope

- SWE-bench Lite as a maintained benchmark or paper result
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

## Coding benchmark status

- The final maintained coding benchmark is `controlled_coding_drift`.
- SWE-bench Lite was attempted as an external agent pilot, but it produced empty patches and no resolved tasks.
- The final paper should not claim SWE-bench improvement.
- The main paper claim is memory correctness under repository drift, not external-task benchmark success.

Synthetic smoke tasks remain only as a maintained local/dev validation path for the OpenRouter-backed executor and table-export workflow.

## Future work

External agent benchmarks such as SWE-bench or LongCoT can be revisited later, but they are not part of the final maintained evaluation.

## OpenRouter defaults

- Base URL env var: `VTM_OPENROUTER_BASE_URL`
- Default base URL: `https://openrouter.ai/api/v1`
- API key env var: `OPENROUTER_API_KEY`
- Execution model env var: `VTM_EXECUTION_MODEL`
- Rerank model env var: `VTM_RERANK_MODEL`
- Smoke/dev rerank model: `nvidia/nemotron-3-nano-30b-a3b:free`
- Default execution model: `google/gemma-4-31b-it:free`
- Optional stronger ablation model: `nvidia/nemotron-3-super-120b-a12b:free`
