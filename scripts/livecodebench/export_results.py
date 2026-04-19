#!/usr/bin/env python3
"""Aggregate LiveCodeBench baseline metadata into paper-table summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Final

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT: Final[Path] = PROJECT_ROOT / ".benchmarks" / "livecodebench"
DEFAULT_OUTPUT_ROOT: Final[Path] = (
    PROJECT_ROOT / ".benchmarks" / "paper-tables" / "livecodebench-baselines"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export LiveCodeBench baseline metadata into paper-table ready summaries."
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--summary-name",
        default="summary",
        help="Base output filename. Writes both <name>.json and <name>.md.",
    )
    return parser


def parse_metadata(path: Path) -> dict[str, str]:
    payload: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        payload[key] = value
    return payload


def collect_rows(input_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if not input_root.exists():
        return rows
    for metadata_path in sorted(input_root.rglob("metadata.txt")):
        row = parse_metadata(metadata_path)
        row["metadata_path"] = str(metadata_path)
        rows.append(row)
    return rows


def write_json(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown(path: Path, rows: list[dict[str, str]]) -> None:
    lines = [
        "# LiveCodeBench Baseline Summary",
        "",
        "LiveCodeBench is reported here as baseline model evaluation only. "
        "It is not a VTM memory-drift result.",
        "",
        "| run_id | model | scenario | release | smoke | evaluate | output_dir |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            (
                "| {run_id} | {model} | {scenario} | {release_version} | "
                "{smoke} | {evaluate} | {output_dir} |"
            ).format(
                run_id=row.get("run_id", ""),
                model=row.get("model", ""),
                scenario=row.get("scenario", ""),
                release_version=row.get("release_version", ""),
                smoke=row.get("smoke", ""),
                evaluate=row.get("evaluate", ""),
                output_dir=row.get("summary_path", row.get("benchmark_root", "")),
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    rows = collect_rows(args.input_root)
    output_json = args.output_root / f"{args.summary_name}.json"
    output_md = args.output_root / f"{args.summary_name}.md"
    write_json(output_json, rows)
    write_markdown(output_md, rows)
    print(f"[livecodebench] wrote {len(rows)} rows to {output_json}")
    print(f"[livecodebench] wrote markdown summary to {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
