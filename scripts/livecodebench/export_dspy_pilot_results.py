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
                "attempt1_pass_at_1": payload.get("attempt1_public_test_pass_at_1"),
                "attempt1_pass_at_k": payload.get("attempt1_public_test_pass_at_k"),
                "attempt1_pass_curve": payload.get("attempt1_public_test_pass_curve", {}),
                "attempt2_pass_at_1": payload.get("attempt2_public_test_pass_at_1"),
                "attempt2_pass_at_k": payload.get("attempt2_public_test_pass_at_k"),
                "attempt2_pass_curve": payload.get("attempt2_public_test_pass_curve", {}),
                "attempt2_delta_over_attempt1": payload.get("attempt2_delta_over_attempt1"),
                "attempt3_pass_at_1": payload.get("attempt3_public_test_pass_at_1"),
                "attempt3_pass_at_k": payload.get("attempt3_public_test_pass_at_k"),
                "attempt3_pass_curve": payload.get("attempt3_public_test_pass_curve", {}),
                "attempt3_delta_over_attempt2": payload.get("attempt3_delta_over_attempt2"),
                "candidate_selection_mode": payload.get("candidate_selection_mode", "single_sample"),
                "candidates_per_attempt": payload.get("candidates_per_attempt", 1),
                "retrieval_usage_rate": payload.get("retrieval_usage_rate", 0),
                "mean_verified_count": payload.get("mean_verified_count", 0),
                "mean_stale_filtered_count": payload.get("mean_stale_filtered_count", 0),
                "mean_tool_calls": payload.get("mean_tool_calls", 0),
                "canonical_memory_hit_rate": payload.get("canonical_memory_hit_rate", 0),
                "mean_canonical_memory_hit_count": payload.get("mean_canonical_memory_hit_count", 0),
                "repair_handoff_hit_rate": payload.get("repair_handoff_hit_rate", 0),
                "repair_handoff_success_rate": payload.get("repair_handoff_success_rate", 0),
                "repair_handoff_card_in_prompt_rate": payload.get("repair_handoff_card_in_prompt_rate", 0),
                "contract_card_in_prompt_rate": payload.get("contract_card_in_prompt_rate", 0),
                "public_test_card_in_prompt_rate": payload.get("public_test_card_in_prompt_rate", 0),
                "attempt2_success_with_canonical_hit_rate": payload.get(
                    "attempt2_success_with_canonical_hit_rate",
                    0,
                ),
                "attempt2_success_without_canonical_hit_rate": payload.get(
                    "attempt2_success_without_canonical_hit_rate",
                    0,
                ),
                "attempt3_success_with_canonical_hit_rate": payload.get(
                    "attempt3_success_with_canonical_hit_rate",
                    0,
                ),
                "attempt3_success_without_canonical_hit_rate": payload.get(
                    "attempt3_success_without_canonical_hit_rate",
                    0,
                ),
                "attempt2_success_when_repair_card_in_prompt": payload.get(
                    "attempt2_success_when_repair_card_in_prompt",
                    0,
                ),
                "attempt2_success_when_no_repair_card_in_prompt": payload.get(
                    "attempt2_success_when_no_repair_card_in_prompt",
                    0,
                ),
                "attempt3_success_when_repair_card_in_prompt": payload.get(
                    "attempt3_success_when_repair_card_in_prompt",
                    0,
                ),
                "attempt3_success_when_no_repair_card_in_prompt": payload.get(
                    "attempt3_success_when_no_repair_card_in_prompt",
                    0,
                ),
                "agent_memory_write_rate": payload.get("agent_memory_write_rate", 0),
                "mean_agent_memory_write_count": payload.get("mean_agent_memory_write_count", 0),
                "total_agent_memory_write_count": payload.get("total_agent_memory_write_count", 0),
                "consolidated_memory_card_rate": payload.get("consolidated_memory_card_rate", 0),
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
        "| Scenario | Method | Model | Problems | Public Test Pass Rate | Attempt1 Pass@1 | Attempt1 Pass@K | Attempt2 Pass@1 | Attempt2 Pass@K | Attempt2 Delta | Attempt3 Pass@1 | Attempt3 Pass@K | Attempt3 Delta | Pass Curve (A1) | Pass Curve (A2) | Pass Curve (A3) | Selection Mode | "
        "Candidates/Attempt | Retrieval Usage Rate | Canonical Hit Rate | Repair Handoff Hit Rate | Handoff In Prompt Rate | Contract In Prompt Rate | Public Test In Prompt Rate | A2 Repair In Prompt Success | A2 No Repair In Prompt Success | A3 Repair In Prompt Success | A3 No Repair In Prompt Success | Agent Write Rate | Mean Agent Writes | Consolidated Card Rate | Mean Verified Count | Mean Stale Filtered Count | Mean Tool Calls |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        pass_rate = "" if row["pass_rate"] is None else f"{float(row['pass_rate']):.3f}"
        attempt1_pass_at_1 = (
            "" if row["attempt1_pass_at_1"] is None else f"{float(row['attempt1_pass_at_1']):.3f}"
        )
        attempt1_pass_at_k = (
            "" if row["attempt1_pass_at_k"] is None else f"{float(row['attempt1_pass_at_k']):.3f}"
        )
        attempt2_pass_at_1 = (
            "" if row["attempt2_pass_at_1"] is None else f"{float(row['attempt2_pass_at_1']):.3f}"
        )
        attempt2_pass_at_k = (
            "" if row["attempt2_pass_at_k"] is None else f"{float(row['attempt2_pass_at_k']):.3f}"
        )
        attempt2_delta = (
            ""
            if row["attempt2_delta_over_attempt1"] is None
            else f"{float(row['attempt2_delta_over_attempt1']):.3f}"
        )
        attempt3_pass_at_1 = (
            "" if row["attempt3_pass_at_1"] is None else f"{float(row['attempt3_pass_at_1']):.3f}"
        )
        attempt3_pass_at_k = (
            "" if row["attempt3_pass_at_k"] is None else f"{float(row['attempt3_pass_at_k']):.3f}"
        )
        attempt3_delta = (
            ""
            if row["attempt3_delta_over_attempt2"] is None
            else f"{float(row['attempt3_delta_over_attempt2']):.3f}"
        )
        lines.append(
            "| {scenario} | {method} | {model} | {problems} | {pass_rate} | {attempt1_pass_at_1} | {attempt1_pass_at_k} | "
            "{attempt2_pass_at_1} | {attempt2_pass_at_k} | {attempt2_delta} | {attempt3_pass_at_1} | {attempt3_pass_at_k} | {attempt3_delta} | "
            "{attempt1_pass_curve} | {attempt2_pass_curve} | {attempt3_pass_curve} | {candidate_selection_mode} | {candidates_per_attempt} | {retrieval_usage_rate:.3f} | "
            "{canonical_memory_hit_rate:.3f} | {repair_handoff_hit_rate:.3f} | {repair_handoff_card_in_prompt_rate:.3f} | {contract_card_in_prompt_rate:.3f} | {public_test_card_in_prompt_rate:.3f} | "
            "{attempt2_success_when_repair_card_in_prompt:.3f} | {attempt2_success_when_no_repair_card_in_prompt:.3f} | "
            "{attempt3_success_when_repair_card_in_prompt:.3f} | {attempt3_success_when_no_repair_card_in_prompt:.3f} | "
            "{agent_memory_write_rate:.3f} | {mean_agent_memory_write_count:.3f} | "
            "{consolidated_memory_card_rate:.3f} | {mean_verified_count:.3f} | {mean_stale_filtered_count:.3f} | {mean_tool_calls:.3f} |".format(
                scenario=row["scenario"],
                method=row["method"],
                model=row["model"],
                problems=row["problems"],
                pass_rate=pass_rate,
                attempt1_pass_at_1=attempt1_pass_at_1,
                attempt1_pass_at_k=attempt1_pass_at_k,
                attempt2_pass_at_1=attempt2_pass_at_1,
                attempt2_pass_at_k=attempt2_pass_at_k,
                attempt2_delta=attempt2_delta,
                attempt3_pass_at_1=attempt3_pass_at_1,
                attempt3_pass_at_k=attempt3_pass_at_k,
                attempt3_delta=attempt3_delta,
                attempt1_pass_curve=_format_pass_curve(row["attempt1_pass_curve"]),
                attempt2_pass_curve=_format_pass_curve(row["attempt2_pass_curve"]),
                attempt3_pass_curve=_format_pass_curve(row["attempt3_pass_curve"]),
                candidate_selection_mode=row["candidate_selection_mode"],
                candidates_per_attempt=int(row["candidates_per_attempt"] or 1),
                retrieval_usage_rate=float(row["retrieval_usage_rate"] or 0),
                canonical_memory_hit_rate=float(row["canonical_memory_hit_rate"] or 0),
                repair_handoff_hit_rate=float(row["repair_handoff_hit_rate"] or 0),
                repair_handoff_card_in_prompt_rate=float(row["repair_handoff_card_in_prompt_rate"] or 0),
                contract_card_in_prompt_rate=float(row["contract_card_in_prompt_rate"] or 0),
                public_test_card_in_prompt_rate=float(row["public_test_card_in_prompt_rate"] or 0),
                attempt2_success_when_repair_card_in_prompt=float(
                    row["attempt2_success_when_repair_card_in_prompt"] or 0
                ),
                attempt2_success_when_no_repair_card_in_prompt=float(
                    row["attempt2_success_when_no_repair_card_in_prompt"] or 0
                ),
                attempt3_success_when_repair_card_in_prompt=float(
                    row["attempt3_success_when_repair_card_in_prompt"] or 0
                ),
                attempt3_success_when_no_repair_card_in_prompt=float(
                    row["attempt3_success_when_no_repair_card_in_prompt"] or 0
                ),
                agent_memory_write_rate=float(row["agent_memory_write_rate"] or 0),
                mean_agent_memory_write_count=float(row["mean_agent_memory_write_count"] or 0),
                consolidated_memory_card_rate=float(row["consolidated_memory_card_rate"] or 0),
                mean_verified_count=float(row["mean_verified_count"] or 0),
                mean_stale_filtered_count=float(row["mean_stale_filtered_count"] or 0),
                mean_tool_calls=float(row["mean_tool_calls"] or 0),
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_pass_curve(curve: Any) -> str:
    if not isinstance(curve, dict):
        return ""
    parts = []
    for key in sorted(curve, key=lambda value: int(str(value)) if str(value).isdigit() else str(value)):
        value = curve.get(key)
        if isinstance(value, int | float):
            parts.append(f"{key}:{float(value):.3f}")
    return ", ".join(parts)


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
