#!/usr/bin/env python3
"""Aggregate LiveCodeBench baseline metadata into paper-table summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
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


def _load_summary_payload(summary_path: Path) -> dict[str, Any]:
    if not summary_path.exists():
        return {}
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def collect_rows(input_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not input_root.exists():
        return rows
    for metadata_path in sorted(input_root.rglob("metadata.txt")):
        row: dict[str, Any] = parse_metadata(metadata_path)
        row["metadata_path"] = str(metadata_path)
        summary_path_value = row.get("summary_path")
        if isinstance(summary_path_value, str) and summary_path_value.strip():
            summary_path = Path(summary_path_value)
            summary_payload = _load_summary_payload(summary_path)
            if summary_payload:
                row.update(summary_payload)
                row["summary_json_path"] = str(summary_path)
        rows.append(row)
    return rows


def write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# LiveCodeBench Baseline Summary",
        "",
        "LiveCodeBench is reported here as baseline model evaluation only. "
        "It is not a VTM memory-drift result.",
        "",
        "| run_id | model | scenario | release | status | pass@1 | pass@5 | evaluate | benchmark_output_dir |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        pass_at_1 = row.get("official_pass_at_1")
        pass_at_5 = row.get("official_pass_at_5")
        lines.append(
            (
                "| {run_id} | {model} | {scenario} | {release_version} | "
                "{status} | {pass_at_1} | {pass_at_5} | {evaluate} | {benchmark_output_dir} |"
            ).format(
                run_id=row.get("run_id", ""),
                model=row.get("model", ""),
                scenario=row.get("scenario", ""),
                release_version=row.get("release_version", ""),
                status=row.get("status", ""),
                pass_at_1="" if pass_at_1 in (None, "") else f"{float(pass_at_1):.4f}",
                pass_at_5="" if pass_at_5 in (None, "") else f"{float(pass_at_5):.4f}",
                evaluate=row.get("evaluate", ""),
                benchmark_output_dir=row.get(
                    "benchmark_output_dir",
                    row.get(
                        "wrapper_output_dir",
                        row.get("output_dir", row.get("summary_path", row.get("benchmark_root", ""))),
                    ),
                ),
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
