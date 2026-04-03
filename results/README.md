# Results Layout

This directory stores benchmark outputs produced by scripts in ../scripts.

## Structure

- raw/livecodebench/<run_id>/
- raw/swebench/<run_id>/
- metrics/
- runs/
- visualizations/

## Conventions

- Keep each benchmark run isolated by run id.
- Include metadata.txt per run for reproducibility.
- Keep command logs with the exact command used.
- Write normalized benchmark summaries to metrics/.

## Benchmark Environments

Set up a local `uv` environment inside each benchmark repository before running the benchmark scripts. The runner scripts will automatically use `benchmarks/<name>/.venv/bin/python` when it exists.

```bash
cd benchmarks/LiveCodeBench
uv venv --python 3.11
uv pip install -e .

cd ../SWE-bench
uv venv --python 3.11
uv pip install -e .
```

## Normalize + Visualize

From the repository root:

```bash
uv venv --python 3.13
source .venv/bin/activate
uv sync --extra results

cd results
python3 normalize_results.py
python3 visualize_results.py --latest-only
```

If you only want the minimum plotting dependency instead of syncing extras:

```bash
uv pip install matplotlib
```

Outputs:

- metrics/normalized_metrics.jsonl
- metrics/normalized_metrics.csv
- metrics/extraction_sources.json
- visualizations/summary.md
- visualizations/*.png (when matplotlib is available)

## Notes

- If raw run folders are empty, normalization still writes empty metrics files.
- If normalized_metrics.csv only contains the header row, there is nothing to plot yet. You need at least one run directory under raw/livecodebench/ or raw/swebench/.
- SWE-bench solve rate is extracted from run report JSON when available, with log parsing fallback.
- LiveCodeBench pass metrics are extracted from *_eval.json outputs discovered from output_files.txt or default output paths.
- Extraction provenance is recorded in metrics/extraction_sources.json so you can see exactly which files were parsed per run.
- Cost and token usage fields are currently nullable placeholders until a stable source is wired in.

## Zero To Graphs

1. Prepare the workspace and benchmark clones:

```bash
scripts/setup_project.sh
scripts/preflight_checks.sh --benchmark all
```

2. Generate raw benchmark outputs. Examples:

```bash
scripts/run_livecodebench_baseline.sh --model gpt-4-1106-preview --scenario codegeneration --evaluate true
scripts/run_swebench_baseline.sh --mode eval-only --predictions-path <path_to_predictions.jsonl>
```

For a fast no-cost SWE-bench smoke test, you can use the built-in gold patches and a single instance id:

```bash
scripts/run_swebench_baseline.sh \
  --mode eval-only \
  --predictions-path gold \
  --instance-id astropy__astropy-12907 \
  --max-workers 1 \
  --timeout 600 \
  --run-id swe_gold_smoke
```

3. Regenerate metrics and charts:

```bash
cd results
python3 normalize_results.py
python3 visualize_results.py --latest-only
```

4. Verify these artifacts exist and are non-empty:

- metrics/normalized_metrics.csv
- metrics/extraction_sources.json
- visualizations/summary.md
- visualizations/*.png
