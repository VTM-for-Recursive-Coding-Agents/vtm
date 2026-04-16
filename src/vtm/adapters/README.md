# src/vtm/adapters

Purpose: provider and environment integration boundary for the kernel, retrieval stack, and optional model backends.

Contents
- `git.py`: Repository fingerprint collection.
- `runtime.py`: Runtime and tool-version fingerprint collection.
- `python_ast.py` and `tree_sitter.py`: Syntax-anchor construction and relocation.
- `embeddings.py`: Deterministic embedding adapter contract and local reference implementation.
- `rlm.py`: Provider-neutral reranking contract.
- `openai_*.py`: Optional OpenAI-compatible reference implementations.
