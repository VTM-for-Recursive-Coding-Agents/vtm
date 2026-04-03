#!/usr/bin/env python3
"""Generate benchmark visualizations from normalized metrics."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

from normalize_results import write_outputs, discover_runs, write_source_map


def _print_no_data_guidance(raw_root: Path, metrics_dir: Path, output_dir: Path) -> None:
	print("[viz] No benchmark rows are available to visualize.")
	print(f"[viz] Expected raw runs under: {raw_root}")
	print(f"[viz] Latest metrics file: {metrics_dir / 'normalized_metrics.csv'}")
	print(f"[viz] Summary written to: {output_dir / 'summary.md'}")
	print("[viz] Next steps:")
	print("  1. Generate at least one benchmark run so results/raw contains run folders.")
	print("  2. Re-run normalize_results.py to populate results/metrics/normalized_metrics.csv.")
	print("  3. Re-run visualize_results.py once the metrics file has data rows.")


def _print_matplotlib_guidance() -> None:
	print("[viz] matplotlib not available; skipping chart rendering.")
	print("[viz] Install plotting support with either:")
	print("  - uv sync --extra results")
	print("  - uv pip install matplotlib")

#this method reads the normalized metrics from a CSV file and returns a list of dictionaries, where each dictionary represents a row of the CSV file with column names as keys. It handles the case where the file does not exist by returning an empty list.
def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
	if not path.exists():
		return []
	with path.open("r", encoding="utf-8", newline="") as handle:
		return list(csv.DictReader(handle))

#this method attempts to convert a given value to a float. It handles various cases, such as when the value is None, an empty string, or the string "none" (case-insensitive), in which case it returns None. If the value is already a float or an integer, it converts it to a float. If the conversion fails due to a ValueError, it also returns None.
def _to_float(value: Any) -> float | None:
	if value is None:
		return None
	if isinstance(value, float):
		return value
	if isinstance(value, int):
		return float(value)
	text = str(value).strip()
	if text == "" or text.lower() == "none":
		return None
	try:
		return float(text)
	except ValueError:
		return None

#this method takes a list of dictionaries (representing rows of benchmark results) and returns a new list of dictionaries that contains only the latest entry for each unique combination of "benchmark" and "model". It groups the rows by these two keys and compares their "timestamp" values to determine which one is the latest. The resulting list will have at most one entry for each benchmark/model pair, representing the most recent run.
def _latest_by_model_and_benchmark(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
	grouped: dict[tuple[str, str], dict[str, Any]] = {}
	for row in rows:
		key = (row.get("benchmark", ""), row.get("model", ""))
		ts = row.get("timestamp") or ""
		existing = grouped.get(key)
		if existing is None or ts > (existing.get("timestamp") or ""):
			grouped[key] = row
	return list(grouped.values())

#this method generates visualizations based on the provided benchmark results. It first checks if the matplotlib library is available for plotting. If it is not available, it prints a message and returns an empty list of artifacts. If matplotlib is available, it creates various charts to compare scores by model for each benchmark, runtime comparisons, and success/failure counts for SWE runs. The generated charts are saved as PNG files in the specified output directory, and the paths to these files are collected in a list of artifacts that is returned at the end.
def _plot_if_available(output_dir: Path, rows: list[dict[str, Any]]) -> list[Path]:
	try:
		import matplotlib.pyplot as plt  # type: ignore[import-not-found]
	except Exception:
		_print_matplotlib_guidance()
		return []

	output_dir.mkdir(parents=True, exist_ok=True)
	artifacts: list[Path] = []

	# Chart 1: Score comparison by model for each benchmark.
	for benchmark in ("livecodebench", "swebench"):
		subset = [r for r in rows if r.get("benchmark") == benchmark]
		if not subset:
			continue
		labels = [r.get("model", "unknown") for r in subset]
		values = [_to_float(r.get("score")) for r in subset]
		points = [(l, v) for l, v in zip(labels, values) if v is not None and not math.isnan(v)]
		if not points:
			continue

		fig, ax = plt.subplots(figsize=(10, 5))
		x_labels = [p[0] for p in points]
		y_vals = [p[1] for p in points]
		ax.bar(x_labels, y_vals)
		ax.set_ylim(0.0, 1.0)
		ax.set_ylabel("Score")
		ax.set_title(f"{benchmark}: score by model")
		ax.tick_params(axis="x", labelrotation=30)
		fig.tight_layout()
		out = output_dir / f"{benchmark}_score_by_model.png"
		fig.savefig(str(out), dpi=160)
		plt.close(fig)
		artifacts.append(out)

	# Chart 2: Runtime comparison (if runtime exists).
	runtime_rows = [r for r in rows if _to_float(r.get("runtime_seconds")) is not None]
	if runtime_rows:
		fig, ax = plt.subplots(figsize=(10, 5))
		labels = [f"{r.get('benchmark')}:{r.get('model')}" for r in runtime_rows]
		values = [_to_float(r.get("runtime_seconds")) for r in runtime_rows]
		ax.bar(labels, [v if v is not None else 0.0 for v in values])
		ax.set_ylabel("Runtime (seconds)")
		ax.set_title("Runtime by benchmark/model")
		ax.tick_params(axis="x", labelrotation=40)
		fig.tight_layout()
		out = output_dir / "runtime_by_model.png"
		fig.savefig(str(out), dpi=160)
		plt.close(fig)
		artifacts.append(out)

	# Chart 3: Success and failure counts for SWE runs.
	swe_rows = [r for r in rows if r.get("benchmark") == "swebench"]
	if swe_rows:
		labels = [r.get("model", "unknown") for r in swe_rows]
		success = [(_to_float(r.get("success_count")) or 0.0) for r in swe_rows]
		failure = [(_to_float(r.get("failure_count")) or 0.0) for r in swe_rows]
		fig, ax = plt.subplots(figsize=(10, 5))
		ax.bar(labels, success, label="resolved")
		ax.bar(labels, failure, bottom=success, label="unresolved")
		ax.set_ylabel("Instance count")
		ax.set_title("SWE-bench outcome counts by model")
		ax.tick_params(axis="x", labelrotation=30)
		ax.legend()
		fig.tight_layout()
		out = output_dir / "swe_outcomes_by_model.png"
		fig.savefig(str(out), dpi=160)
		plt.close(fig)
		artifacts.append(out)

	return artifacts

#this method generates a summary markdown file that includes a table of benchmark results. It takes a list of dictionaries (representing rows of benchmark results) and an output directory path. The method creates the output directory if it does not exist, constructs a markdown string with a header and a table of the results, and writes this string to a file named "summary.md" in the output directory. Finally, it returns the path to the generated summary file.
def _write_summary(rows: list[dict[str, Any]], output_dir: Path) -> Path:
	output_dir.mkdir(parents=True, exist_ok=True)
	summary_path = output_dir / "summary.md"

	lines = [
		"# Benchmark Visualization Summary",
		"",
		f"Rows considered: {len(rows)}",
		"",
		"| benchmark | model | score | solve_rate | pass_at_1 | pass_at_5 | status |",
		"|---|---|---:|---:|---:|---:|---|",
	]
	for row in rows:
		lines.append(
			"| {benchmark} | {model} | {score} | {solve_rate} | {pass_at_1} | {pass_at_5} | {status} |".format(
				benchmark=row.get("benchmark", ""),
				model=row.get("model", ""),
				score=row.get("score", ""),
				solve_rate=row.get("solve_rate", ""),
				pass_at_1=row.get("pass_at_1", ""),
				pass_at_5=row.get("pass_at_5", ""),
				status=row.get("status", ""),
			)
		)

	summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
	return summary_path

#this method builds an argument parser for the script. It defines several command-line arguments that can be used to specify the input and output directories, filter by model, and control whether to keep only the latest runs or skip normalization. The parser is configured with appropriate types, default values, and help messages for each argument. Finally, the method returns the configured argument parser.
def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Generate benchmark visualizations")
	parser.add_argument(
		"--raw-root",
		type=Path,
		default=Path(__file__).resolve().parent / "raw",
		help="Root folder containing raw benchmark runs.",
	)
	parser.add_argument(
		"--metrics-dir",
		type=Path,
		default=Path(__file__).resolve().parent / "metrics",
		help="Folder containing normalized metrics files.",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=Path(__file__).resolve().parent / "visualizations",
		help="Folder for visualization artifacts.",
	)
	parser.add_argument(
		"--model",
		type=str,
		default="",
		help="Optional model filter.",
	)
	parser.add_argument(
		"--latest-only",
		action="store_true",
		help="Keep only the latest run for each benchmark/model pair.",
	)
	parser.add_argument(
		"--skip-normalize",
		action="store_true",
		help="Skip normalizing raw runs before plotting.",
	)
	return parser


def main() -> int:
	args = build_arg_parser().parse_args()
	if not args.skip_normalize:
		records = discover_runs(args.raw_root)
		write_outputs(records, args.metrics_dir)
		write_source_map(records, args.metrics_dir)
		print(f"[viz] Normalized {len(records)} runs into {args.metrics_dir}")
	rows = _read_csv_rows(args.metrics_dir / "normalized_metrics.csv")

	if args.model:
		rows = [r for r in rows if r.get("model") == args.model]

	if args.latest_only:
		rows = _latest_by_model_and_benchmark(rows)

	summary_path = _write_summary(rows, args.output_dir)
	if not rows:
		_print_no_data_guidance(args.raw_root, args.metrics_dir, args.output_dir)
		print(f"[viz] Summary: {summary_path}")
		print("[viz] No chart images generated.")
		return 0

	artifacts = _plot_if_available(args.output_dir, rows)

	print(f"[viz] Summary: {summary_path}")
	if artifacts:
		print("[viz] Charts:")
		for artifact in artifacts:
			print(f"  - {artifact}")
	else:
		print("[viz] No chart images generated.")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
