# Contributing

## Development setup

Target runtime: Python 3.12.

```bash
uv sync --dev --all-extras
```

## Local quality gate

Run the same checks expected in CI before sending changes for review:

```bash
uv run python -m ruff check .
uv run python -m mypy src
uv run pytest -q
```

## Documentation expectations

Documentation is expected to move with behavior changes.

If a change affects a public contract, file layout, artifact layout, CLI surface, or durable example, update:

- `README.md` when the change is user-facing
- the relevant source-of-truth doc under `docs/`
- the affected package README
- the relevant ADR when the boundary or policy is durable

## Pull requests

- Keep changes scoped and reviewable.
- Add or update tests for behavioral changes.
- Regenerate derived docs when needed, including `docs/code-reference.md`.
- Do not silently change public import boundaries or benchmark artifact layouts without updating docs.
