# src/vtm/adapters

Purpose: provider and environment integration boundary for the kernel, retrieval stack, and optional model backends.

Contents
- `git.py`: Repository fingerprint collection.
- `runtime.py`: Runtime and tool-version fingerprint collection.
- `python_ast.py` and `tree_sitter.py`: Syntax-anchor construction and relocation.
- `openai_chat.py`: thin OpenAI-compatible chat client used for OpenRouter-backed flows.
