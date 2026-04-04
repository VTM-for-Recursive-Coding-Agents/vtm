# src/vtm/adapters

Purpose: external integration boundary for VTM. Adapters collect fingerprints, build anchors, or call model providers without leaking provider-specific logic into the core kernel.

Contents
- `__init__.py`: Re-exports the adapter protocols and concrete implementations.
- `embeddings.py`: Embedding adapter protocol plus the deterministic hash embedding adapter used for local and CI-safe retrieval.
- `git.py`: Git-backed repository fingerprint collector covering branch, commit, tree, diff, and untracked-file state.
- `openai_embedding.py`: Optional OpenAI reference implementation of the embedding adapter.
- `openai_rlm.py`: Optional OpenAI reference implementation of the reranking adapter using structured Responses API output.
- `python_ast.py`: Python AST anchor builder and relocator used as the fallback and parity path.
- `rlm.py`: Provider-neutral reranking request, response, candidate, and adapter protocol definitions.
- `runtime.py`: Runtime and tool-version fingerprint collectors plus the default tool probe set.
- `tree_sitter.py`: Python Tree-sitter anchor adapter with AST fallback and an unavailable stub when Tree-sitter extras are missing.
