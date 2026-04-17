# src/vtm/adapters

Purpose: provider and environment integration boundary for the kernel, retrieval stack, and optional model backends.

Contents
- `git.py`: Repository fingerprint collection.
- `runtime.py`: Runtime and tool-version fingerprint collection.
- `python_ast.py` and `tree_sitter.py`: Syntax-anchor construction and relocation.
- `rlm.py`: Provider-neutral reranking contract.
- `openai_chat.py` and `openai_rlm.py`: thin OpenAI-compatible reference implementations maintained for OpenRouter-backed flows.
