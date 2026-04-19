# LongCoT Pilot

This runner is an optional external reasoning pilot. It does not change VTM retrieval scoring, drift scoring, drifted-retrieval scoring, verifier behavior, or the maintained paper-table exporter.

## Install LongCoT

Clone LongCoT next to the VTM repo or under a local vendor directory, then install it into the current VTM environment:

```bash
git clone https://github.com/LongHorizonReasoning/longcot.git .vendor/longcot
uv pip install -e ./.vendor/longcot
```

LongCoT upstream: <https://github.com/LongHorizonReasoning/longcot>

## Required environment

```bash
export OPENROUTER_API_KEY=...
export VTM_OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
export VTM_EXECUTION_MODEL=google/gemma-4-31b-it:free
```

## Run a 3-question LongCoT-Mini CS pilot

`easy` is the LongCoT-Mini subset. This pilot calls OpenRouter once per question, verifies with `longcot.verify(...)`, and writes `responses.jsonl`, `summary.json`, and `paper_table.md`.

```bash
uv run python scripts/run_longcot_pilot.py \
  --domain cs \
  --difficulty easy \
  --max-questions 3 \
  --output-dir .benchmarks/longcot-pilot-cs-easy-3
```

## Generated artifacts

- `.benchmarks/.../responses.jsonl`: per-question raw response rows plus verification status
- `.benchmarks/.../summary.json`: aggregate counts and accuracy metrics
- `.benchmarks/.../paper_table.md`: a small Markdown table suitable for the paper appendix or external-pilot note

## Interpreting the result

- Treat this as optional external reasoning evidence only.
- Do not mix it into the main VTM retrieval, drift, or drifted-retrieval claims.
- Use it as a deterministic long-horizon reasoning pilot when you want a small non-patch benchmark alongside the main retrieval and controlled coding results.
