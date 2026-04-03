#!/usr/bin/env python3
"""Normalize benchmark run artifacts into a unified metrics table.

This script scans results/raw/{livecodebench,swebench} run folders and emits:
- results/metrics/normalized_metrics.jsonl
- results/metrics/normalized_metrics.csv

It is resilient to partial or missing artifacts and records warnings per run.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class RunRecord:
    benchmark: str
    run_id: str
    model: str
    provider: str
    scenario: str | None
    timestamp: str | None
    status: str
    pass_at_1: float | None
    pass_at_5: float | None
    score: float | None
    solve_rate: float | None
    resolved_instances: int | None
    completed_instances: int | None
    total_instances: int | None
    success_count: int | None
    failure_count: int | None
    error_count: int | None
    runtime_seconds: float | None
    token_usage: float | None
    cost_usd: float | None
    source_path: str
    source_files: list[str]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "run_id": self.run_id,
            "model": self.model,
            "provider": self.provider,
            "scenario": self.scenario,
            "timestamp": self.timestamp,
            "status": self.status,
            "pass_at_1": self.pass_at_1,
            "pass_at_5": self.pass_at_5,
            "score": self.score,
            "solve_rate": self.solve_rate,
            "resolved_instances": self.resolved_instances,
            "completed_instances": self.completed_instances,
            "total_instances": self.total_instances,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "error_count": self.error_count,
            "runtime_seconds": self.runtime_seconds,
            "token_usage": self.token_usage,
            "cost_usd": self.cost_usd,
            "source_path": self.source_path,
            "source_files": self.source_files,
            "warnings": self.warnings,
        }


def _read_kv_metadata(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    if not path.exists():
        return data
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key.strip()] = value.strip()
    return data


def _extract_first_float(pattern: str, text: str) -> float | None:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _extract_first_int(pattern: str, text: str) -> int | None:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _parse_iso_utc(value: str | None) -> str | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return dt.astimezone(timezone.utc).isoformat()
    except ValueError:
        return value


def _pick_lcb_eval_json(run_dir: Path, metadata: dict[str, str]) -> Path | None:
    output_files_path = run_dir / "output_files.txt"
    if output_files_path.exists():
        candidates: list[Path] = []
        for line in output_files_path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            if line.endswith("_eval.json"):
                p = Path(line)
                candidates.append(p)
        if candidates:
            candidates.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0.0, reverse=True)
            for c in candidates:
                if c.exists():
                    return c

    model = metadata.get("model", "")
    scenario = metadata.get("scenario", "")
    n = metadata.get("n", "")
    temp = metadata.get("temperature", "")
    root = run_dir.parents[2] if len(run_dir.parents) >= 3 else run_dir
    lcb_output = root / "benchmarks" / "LiveCodeBench" / "output"
    if model and scenario and n and temp and lcb_output.exists():
        guess = lcb_output / model / f"{scenario}_{n}_{temp}_eval.json"
        if guess.exists():
            return guess
    return None


def _parse_lcb_metrics(eval_json: Path) -> tuple[float | None, float | None, float | None, list[str]]:
    warnings: list[str] = []
    try:
        payload = json.loads(eval_json.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, None, None, [f"failed_to_read_lcb_eval_json:{exc}"]

    pass1 = None
    pass5 = None
    score = None

    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        first = payload[0]
        pass1 = first.get("pass@1")
        if pass1 is not None:
            try:
                pass1 = float(pass1)
            except (TypeError, ValueError):
                pass1 = None

        pass5 = first.get("pass@5")
        if pass5 is not None:
            try:
                pass5 = float(pass5)
            except (TypeError, ValueError):
                pass5 = None

        if pass1 is not None:
            score = pass1
    else:
        warnings.append("unexpected_lcb_eval_json_schema")

    return pass1, pass5, score, warnings


def parse_livecodebench_run(run_dir: Path) -> RunRecord:
    metadata = _read_kv_metadata(run_dir / "metadata.txt")
    warnings: list[str] = []

    model = metadata.get("model", "unknown")
    run_id = metadata.get("run_id", run_dir.name)
    timestamp = _parse_iso_utc(metadata.get("started_at"))
    scenario = metadata.get("scenario")

    pass1 = None
    pass5 = None
    score = None
    source_files: list[str] = []

    metadata_path = run_dir / "metadata.txt"
    if metadata_path.exists():
        source_files.append(str(metadata_path))

    eval_json = _pick_lcb_eval_json(run_dir, metadata)
    if eval_json is not None:
        pass1, pass5, score, metric_warnings = _parse_lcb_metrics(eval_json)
        warnings.extend(metric_warnings)
        source_files.append(str(eval_json))
    else:
        warnings.append("lcb_eval_json_not_found")

    output_files_path = run_dir / "output_files.txt"
    if output_files_path.exists():
        source_files.append(str(output_files_path))

    runtime_seconds = None
    command_log = run_dir / "command.log"
    if command_log.exists():
        source_files.append(str(command_log))
        text = command_log.read_text(encoding="utf-8", errors="replace")
        runtime_seconds = _extract_first_float(r"runtime[^0-9]*([0-9]+(?:\.[0-9]+)?)", text)

    status = "success" if (pass1 is not None or pass5 is not None) else "partial"

    return RunRecord(
        benchmark="livecodebench",
        run_id=run_id,
        model=model,
        provider="baseline",
        scenario=scenario,
        timestamp=timestamp,
        status=status,
        pass_at_1=pass1,
        pass_at_5=pass5,
        score=score,
        solve_rate=None,
        resolved_instances=None,
        completed_instances=None,
        total_instances=None,
        success_count=1 if status == "success" else 0,
        failure_count=0 if status == "success" else 1,
        error_count=0,
        runtime_seconds=runtime_seconds,
        token_usage=None,
        cost_usd=None,
        source_path=str(run_dir),
        source_files=source_files,
        warnings=warnings,
    )


def _find_swe_report_json(run_dir: Path, run_id: str, model: str) -> Path | None:
    local_report = run_dir / f"{model.replace('/', '__')}.{run_id}.json"
    if local_report.exists():
        return local_report

    root = run_dir.parents[2] if len(run_dir.parents) >= 3 else run_dir
    swe_root = root / "benchmarks" / "SWE-bench"
    repo_report = swe_root / f"{model.replace('/', '__')}.{run_id}.json"
    if repo_report.exists():
        return repo_report

    matches = list(swe_root.glob(f"*.{run_id}.json")) if swe_root.exists() else []
    return matches[0] if matches else None


def _parse_swe_from_log(text: str) -> tuple[int | None, int | None, int | None, int | None, list[str]]:
    warnings: list[str] = []
    total = _extract_first_int(r"Total instances:\s*(\d+)", text)
    completed = _extract_first_int(r"Instances completed:\s*(\d+)", text)
    resolved = _extract_first_int(r"Instances resolved:\s*(\d+)", text)
    errors = _extract_first_int(r"Instances with errors:\s*(\d+)", text)
    if total is None and completed is None and resolved is None:
        warnings.append("swe_summary_not_found_in_log")
    return total, completed, resolved, errors, warnings


def parse_swebench_run(run_dir: Path) -> RunRecord:
    metadata = _read_kv_metadata(run_dir / "metadata.txt")
    warnings: list[str] = []

    model = metadata.get("model", "unknown")
    run_id = metadata.get("run_id", run_dir.name)
    timestamp = _parse_iso_utc(metadata.get("started_at"))
    scenario = metadata.get("mode")

    total = None
    completed = None
    resolved = None
    errors = None
    source_files: list[str] = []

    metadata_path = run_dir / "metadata.txt"
    if metadata_path.exists():
        source_files.append(str(metadata_path))

    report_json = _find_swe_report_json(run_dir, run_id, model)
    if report_json and report_json.exists():
        try:
            report = json.loads(report_json.read_text(encoding="utf-8"))
            total = report.get("total_instances")
            completed = report.get("completed_instances")
            resolved = report.get("resolved_instances")
            errors = report.get("error_instances")
            source_files.append(str(report_json))
        except Exception as exc:
            warnings.append(f"failed_to_read_swe_report_json:{exc}")

    if total is None or completed is None or resolved is None:
        log_path = run_dir / "evaluation.log"
        if log_path.exists():
            source_files.append(str(log_path))
            log_text = log_path.read_text(encoding="utf-8", errors="replace")
            log_total, log_completed, log_resolved, log_errors, log_warnings = _parse_swe_from_log(log_text)
            warnings.extend(log_warnings)
            total = total if total is not None else log_total
            completed = completed if completed is not None else log_completed
            resolved = resolved if resolved is not None else log_resolved
            errors = errors if errors is not None else log_errors
        else:
            warnings.append("swe_evaluation_log_not_found")

    solve_rate = None
    if resolved is not None and completed not in (None, 0):
        solve_rate = float(resolved) / float(completed)

    status = "success" if solve_rate is not None else "partial"
    failure_count = None
    if completed is not None and resolved is not None:
        failure_count = max(completed - resolved, 0)

    return RunRecord(
        benchmark="swebench",
        run_id=run_id,
        model=model,
        provider="baseline",
        scenario=scenario,
        timestamp=timestamp,
        status=status,
        pass_at_1=solve_rate,
        pass_at_5=None,
        score=solve_rate,
        solve_rate=solve_rate,
        resolved_instances=resolved,
        completed_instances=completed,
        total_instances=total,
        success_count=resolved,
        failure_count=failure_count,
        error_count=errors,
        runtime_seconds=None,
        token_usage=None,
        cost_usd=None,
        source_path=str(run_dir),
        source_files=source_files,
        warnings=warnings,
    )


def discover_runs(raw_root: Path) -> list[RunRecord]:
    runs: list[RunRecord] = []

    lcb_root = raw_root / "livecodebench"
    if lcb_root.exists():
        for run_dir in sorted([p for p in lcb_root.iterdir() if p.is_dir()]):
            runs.append(parse_livecodebench_run(run_dir))

    swe_root = raw_root / "swebench"
    if swe_root.exists():
        for run_dir in sorted([p for p in swe_root.iterdir() if p.is_dir()]):
            runs.append(parse_swebench_run(run_dir))

    return runs


def write_outputs(records: list[RunRecord], metrics_dir: Path) -> tuple[Path, Path]:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = metrics_dir / "normalized_metrics.jsonl"
    csv_path = metrics_dir / "normalized_metrics.csv"

    row_dicts = [r.to_dict() for r in records]

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in row_dicts:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    fieldnames = [
        "benchmark",
        "run_id",
        "model",
        "provider",
        "scenario",
        "timestamp",
        "status",
        "pass_at_1",
        "pass_at_5",
        "score",
        "solve_rate",
        "resolved_instances",
        "completed_instances",
        "total_instances",
        "success_count",
        "failure_count",
        "error_count",
        "runtime_seconds",
        "token_usage",
        "cost_usd",
        "source_path",
        "source_files",
        "warnings",
    ]

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in row_dicts:
            row_copy = dict(row)
            row_copy["source_files"] = ";".join(row_copy["source_files"])
            row_copy["warnings"] = ";".join(row_copy["warnings"])
            writer.writerow(row_copy)

    return jsonl_path, csv_path


def write_source_map(records: list[RunRecord], metrics_dir: Path) -> Path:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    source_map_path = metrics_dir / "extraction_sources.json"
    payload = [
        {
            "benchmark": r.benchmark,
            "run_id": r.run_id,
            "model": r.model,
            "source_path": r.source_path,
            "source_files": r.source_files,
            "warnings": r.warnings,
        }
        for r in records
    ]
    source_map_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return source_map_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Normalize benchmark run artifacts")
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path(__file__).resolve().parent / "raw",
        help="Root folder containing livecodebench/ and swebench/ run folders.",
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "metrics",
        help="Output folder for normalized metrics files.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    records = discover_runs(args.raw_root)
    jsonl_path, csv_path = write_outputs(records, args.metrics_dir)
    source_map_path = write_source_map(records, args.metrics_dir)

    print(f"[normalize] Runs processed: {len(records)}")
    print(f"[normalize] JSONL: {jsonl_path}")
    print(f"[normalize] CSV:   {csv_path}")
    print(f"[normalize] Sources: {source_map_path}")

    warning_count = sum(1 for r in records if r.warnings)
    print(f"[normalize] Runs with warnings: {warning_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
