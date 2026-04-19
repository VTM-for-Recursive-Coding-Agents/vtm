# Results Layout

This directory stores benchmark outputs and derived analysis artifacts only.

## Structure

- raw/livecodebench/<run_id>/
- raw/swebench/<run_id>/
- metrics/
- visualizations/

Generated launcher bundles no longer belong here. Use `../launchers/` for local and CHPC launcher bundles.

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

To keep unusable runs out of analysis artifacts, use the explicit exclusion flag:

```bash
cd results
python3 normalize_results.py --exclude-unusable
python3 visualize_results.py --latest-only --exclude-unusable
```

To audit or delete unusable raw and archived run folders:

```bash
cd results
python3 prune_unusable_runs.py
python3 prune_unusable_runs.py --delete
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

## Pass/Fail Dashboard

To graph method-level pass/fail results and current rlmfix checkpoint health:

```bash
cd results
python3 plot_passfail_graphs.py
```

To compare baseline, rag, rlm, and rlm_rag on only the questions all four methods answered:

```bash
cd results
python3 plot_passfail_graphs.py \
  --intersection-only \
  --intersection-model-prefix Qwen2.5-Coder-Ins-32B-
```

If both partial and non-partial TSVs exist for a method, you can choose source preference:

```bash
cd results
python3 plot_passfail_graphs.py \
  --intersection-only \
  --intersection-model-prefix Qwen2.5-Coder-Ins-32B- \
  --intersection-partial-policy prefer_partial
```

This reads:

- `results/runs/*passfail.tsv`
- `results/raw/livecodebench/*/rlm_progress.jsonl`

And writes:

- `results/visualizations/passfail_dashboard.png`
- `results/visualizations/passfail_dashboard_summary.md`

In intersection mode, the summary includes:

- shared-question count across `baseline/rag/rlm/rlm_rag`
- pass@1 rate per method on that shared subset
- raw pass/fail counts per method on that shared subset

## Notes

- If raw run folders are empty, normalization still writes empty metrics files.
- If normalized_metrics.csv only contains the header row, there is nothing to plot yet. You need at least one run directory under raw/livecodebench/ or raw/swebench/.
- SWE-bench solve rate is extracted from run report JSON when available, with log parsing fallback.
- LiveCodeBench pass metrics are extracted from *_eval.json outputs discovered from output_files.txt or default output paths.
- Extraction provenance is recorded in metrics/extraction_sources.json so you can see exactly which files were parsed per run.
- Cost and token usage fields are currently nullable placeholders until a stable source is wired in.
- A run is considered unusable for analysis when it is not successful, has warnings, or is explicitly named with --known-failed-run-id.
- prune_unusable_runs.py is dry-run by default; pass --delete only after reviewing the candidate list.

## Zero To Graphs

1. Prepare the workspace and benchmark clones:

```bash
scripts/setup_project.sh
scripts/local/preflight_checks.sh --benchmark all
```

2. Generate raw benchmark outputs. Examples:

```bash
scripts/local/run_livecodebench.sh --model gpt-4-1106-preview --scenario codegeneration --evaluate true
scripts/local/run_swebench.sh --mode eval-only --predictions-path <path_to_predictions.jsonl>
```

For a fast no-cost SWE-bench smoke test, you can use the built-in gold patches and a single instance id:

```bash
scripts/local/run_swebench.sh \
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
python3 normalize_results.py --exclude-unusable
python3 visualize_results.py --latest-only --exclude-unusable
```

4. Verify these artifacts exist and are non-empty:

- metrics/normalized_metrics.csv
- metrics/extraction_sources.json
- visualizations/summary.md
- visualizations/*.png
