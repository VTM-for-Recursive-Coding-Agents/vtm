#!/usr/bin/env python3
"""Export per-question pass/fail rows from LiveCodeBench evaluation files.

Supports both file formats produced by LiveCodeBench:
- *_eval_all.json (preferred): list[dict], each with question_id and pass@1
- *_eval.json: list where metrics[0]["detail"]["pass@1"] maps question_id -> score

Examples:
  python3 results/export_lcb_passfail.py \
    --input benchmarks/LiveCodeBench/output/Qwen2.5-Coder-Ins-32B-rlm/Scenario.codegeneration_1_0.2_codegeneration_output_eval_all.json \
    --output /tmp/rlm_passfail.tsv

  python3 results/export_lcb_passfail.py \
    --input benchmarks/LiveCodeBench/output/Qwen2.5-Coder-Ins-32B-rlm/Scenario.codegeneration_1_0.2_codegeneration_output_eval.json \
    --output /tmp/rlm_passfail.csv --format csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export per-question pass/fail from LCB eval files")
    parser.add_argument("--input", required=True, help="Path to *_eval_all.json or *_eval.json")
    parser.add_argument("--output", required=True, help="Destination .tsv or .csv")
    parser.add_argument(
        "--format",
        choices=["tsv", "csv"],
        default="tsv",
        help="Output delimiter format (default: tsv)",
    )
    parser.add_argument(
        "--pass-threshold",
        type=float,
        default=1.0,
        help="Threshold for PASS label from pass@1 score (default: 1.0)",
    )
    return parser.parse_args()


def _rows_from_eval_all(payload: object) -> list[tuple[str, float]]:
    if not isinstance(payload, list):
        raise ValueError("Expected list payload for *_eval_all.json")

    rows: list[tuple[str, float]] = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        question_id = entry.get("question_id")
        score = entry.get("pass@1")
        if question_id is None or score is None:
            continue
        rows.append((str(question_id), float(score)))
    return rows


def _rows_from_eval(payload: object) -> list[tuple[str, float]]:
    if not isinstance(payload, list) or not payload:
        raise ValueError("Expected non-empty list payload for *_eval.json")

    top = payload[0]
    if not isinstance(top, dict):
        raise ValueError("Unexpected *_eval.json structure (metrics[0] should be a dict)")

    detail = top.get("detail")
    if not isinstance(detail, dict):
        raise ValueError("Missing metrics[0]['detail'] in *_eval.json")

    pass1 = detail.get("pass@1")
    if not isinstance(pass1, dict):
        raise ValueError("Missing metrics[0]['detail']['pass@1'] map in *_eval.json")

    return [(str(question_id), float(score)) for question_id, score in pass1.items()]


def _detect_rows(path: Path, payload: object) -> list[tuple[str, float]]:
    name = path.name
    if name.endswith("_eval_all.json"):
        return _rows_from_eval_all(payload)
    if name.endswith("_eval.json"):
        return _rows_from_eval(payload)

    # Fallback detection by shape if filename is non-standard.
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        first = payload[0]
        if "question_id" in first and "pass@1" in first:
            return _rows_from_eval_all(payload)
        if "detail" in first:
            return _rows_from_eval(payload)

    raise ValueError(
        "Could not detect evaluation file format. Expected *_eval_all.json or *_eval.json structure."
    )


def _write_rows(
    rows: Iterable[tuple[str, float]],
    output: Path,
    fmt: str,
    pass_threshold: float,
) -> None:
    delimiter = "\t" if fmt == "tsv" else ","
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=delimiter)
        writer.writerow(["question_id", "pass_fail", "pass_at_1"])
        for question_id, score in sorted(rows, key=lambda item: item[0]):
            status = "PASS" if score >= pass_threshold else "FAIL"
            writer.writerow([question_id, status, f"{score:.6f}"])


def main() -> int:
    args = _parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    rows = _detect_rows(input_path, payload)
    _write_rows(rows, output_path, args.format, args.pass_threshold)

    print(f"Wrote {len(rows)} rows -> {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
