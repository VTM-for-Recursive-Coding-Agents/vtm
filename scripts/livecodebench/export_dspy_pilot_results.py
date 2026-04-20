#!/usr/bin/env python3
"""Export LiveCodeBench DSPy pilot summaries into a small paper-table markdown file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Final

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT: Final[Path] = PROJECT_ROOT / ".benchmarks" / "livecodebench-dspy"
DEFAULT_OUTPUT_ROOT: Final[Path] = (
    PROJECT_ROOT / ".benchmarks" / "paper-tables" / "livecodebench-dspy-pilot"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export LiveCodeBench DSPy pilot summaries into paper-table markdown."
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def collect_rows(input_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not input_root.exists():
        return rows
    for summary_path in sorted(input_root.rglob("summary.json")):
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        if payload.get("kind") != "dspy_pilot":
            continue
        rows.append(
            {
                "scenario": payload.get("scenario", ""),
                "method": payload.get("method", ""),
                "model": payload.get("model", ""),
                "problems": payload.get("total", 0),
                "pass_rate": payload.get("public_test_pass_rate", payload.get("pass_rate")),
                "retrieval_usage_rate": payload.get("retrieval_usage_rate", 0),
                "mean_verified_count": payload.get("mean_verified_count", 0),
                "mean_stale_filtered_count": payload.get("mean_stale_filtered_count", 0),
                "mean_tool_calls": payload.get("mean_tool_calls", 0),
                "summary_path": str(summary_path),
            }
        )
    return rows


def write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# LiveCodeBench DSPy Pilot",
        "",
        "LiveCodeBench is an external coding benchmark. "
        "This DSPy plus VTM comparison is a scaffolded pilot, not a maintained VTM "
        "retrieval or drift benchmark.",
        "",
        "| Scenario | Method | Model | Problems | Public Test Pass Rate | Retrieval Usage Rate | "
        "Mean Verified Count | Mean Stale Filtered Count | Mean Tool Calls |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        pass_rate = "" if row["pass_rate"] is None else f"{float(row['pass_rate']):.3f}"
        lines.append(
            "| {scenario} | {method} | {model} | {problems} | {pass_rate} | "
            "{retrieval_usage_rate:.3f} | {mean_verified_count:.3f} | "
            "{mean_stale_filtered_count:.3f} | {mean_tool_calls:.3f} |".format(
                scenario=row["scenario"],
                method=row["method"],
                model=row["model"],
                problems=row["problems"],
                pass_rate=pass_rate,
                retrieval_usage_rate=float(row["retrieval_usage_rate"] or 0),
                mean_verified_count=float(row["mean_verified_count"] or 0),
                mean_stale_filtered_count=float(row["mean_stale_filtered_count"] or 0),
                mean_tool_calls=float(row["mean_tool_calls"] or 0),
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    rows = collect_rows(args.input_root)
    output_json = args.output_root / "summary.json"
    output_md = args.output_root / "paper_tables.md"
    write_json(output_json, rows)
    write_markdown(output_md, rows)
    print(f"[livecodebench-dspy-pilot] wrote {len(rows)} rows to {output_json}")
    print(f"[livecodebench-dspy-pilot] wrote markdown summary to {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
