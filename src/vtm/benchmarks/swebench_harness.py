"""Integration helpers for the external SWE-bench harness."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vtm.benchmarks.models import BenchmarkCaseResult, BenchmarkRunConfig, CodingTaskCase


@dataclass(frozen=True)
class SWEbenchHarnessInstanceResult:
    """Normalized SWE-bench evaluation outcome for one instance."""

    instance_id: str
    resolved: bool
    patch_applied: bool
    harness_status: str
    evaluation_log_path: str | None = None


@dataclass(frozen=True)
class SWEbenchHarnessRunArtifacts:
    """Artifact paths produced by one SWE-bench harness run."""

    predictions_path: str
    results_path: str
    logs_dir: str
    stdout_path: str
    stderr_path: str


class SWEbenchHarnessRunner:
    """Writes predictions and normalizes results from the SWE-bench harness."""

    def write_predictions(
        self,
        *,
        cases: list[CodingTaskCase],
        results: list[BenchmarkCaseResult],
        output_dir: Path,
        model_name_or_path: str,
    ) -> Path:
        """Write harness prediction JSONL from local case results."""
        result_map = {result.case_id: result for result in results}
        predictions_path = output_dir / "predictions.jsonl"
        with predictions_path.open("w", encoding="utf-8") as handle:
            for case in cases:
                result = result_map[case.case_id]
                handle.write(
                    json.dumps(
                        {
                            "instance_id": case.instance_id or case.case_id,
                            "model_name_or_path": model_name_or_path,
                            "model_patch": str(result.metadata.get("produced_patch_text", "")),
                        },
                        sort_keys=True,
                    )
                )
                handle.write("\n")
        return predictions_path

    def evaluate_predictions(
        self,
        *,
        cases: list[CodingTaskCase],
        results: list[BenchmarkCaseResult],
        config: BenchmarkRunConfig,
        output_dir: Path,
    ) -> tuple[dict[str, SWEbenchHarnessInstanceResult], SWEbenchHarnessRunArtifacts]:
        """Run the external harness and normalize the resulting report."""
        output_dir = output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        logs_dir = output_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = output_dir / "harness.stdout"
        stderr_path = output_dir / "harness.stderr"

        dataset_name = config.swebench_dataset_name or cases[0].dataset_name
        if not dataset_name:
            raise ValueError("SWE-bench harness tasks require dataset_name")
        predictions_path = self.write_predictions(
            cases=cases,
            results=results,
            output_dir=output_dir,
            model_name_or_path=os.getenv("VTM_LOCAL_LLM_MODEL", "vtm-local-model"),
        )

        run_id = config.swebench_harness_run_id or f"vtm-{config.mode}"
        instance_ids = [case.instance_id or case.case_id for case in cases]
        command = [
            *self._command_prefix(),
            "--dataset_name",
            dataset_name,
            "--split",
            "test",
            "--predictions_path",
            str(predictions_path.resolve()),
            "--max_workers",
            str(config.swebench_harness_workers),
            "--cache_level",
            config.swebench_harness_cache_level,
            "--run_id",
            run_id,
            "--instance_ids",
            ",".join(instance_ids),
            "--report_dir",
            ".",
        ]
        completed = subprocess.run(
            command,
            cwd=output_dir,
            check=False,
            capture_output=True,
            text=True,
        )
        stdout_path.write_text(completed.stdout, encoding="utf-8")
        stderr_path.write_text(completed.stderr, encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(
                "SWE-bench harness evaluation failed; "
                f"see {stderr_path}"
            )

        normalized = self._parse_run_report(
            output_dir=output_dir,
            run_id=run_id,
            instance_ids=instance_ids,
        )
        normalized_path = output_dir / "swebench_harness_results.json"
        normalized_path.write_text(
            json.dumps(
                {
                    instance_id: {
                        "resolved": result.resolved,
                        "patch_applied": result.patch_applied,
                        "harness_status": result.harness_status,
                        "evaluation_log_path": result.evaluation_log_path,
                    }
                    for instance_id, result in normalized.items()
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        artifacts = SWEbenchHarnessRunArtifacts(
            predictions_path=str(predictions_path),
            results_path=str(normalized_path),
            logs_dir=str(logs_dir),
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
        )
        return normalized, artifacts

    def _command_prefix(self) -> tuple[str, ...]:
        override = os.getenv("VTM_SWEBENCH_HARNESS_COMMAND", "").strip()
        if override:
            return tuple(shlex.split(override))
        return (sys.executable, "-m", "swebench.harness.run_evaluation")

    def _parse_run_report(
        self,
        *,
        output_dir: Path,
        run_id: str,
        instance_ids: list[str],
    ) -> dict[str, SWEbenchHarnessInstanceResult]:
        report_payload = self._load_report_payload(output_dir=output_dir, run_id=run_id)
        if report_payload is None:
            return {
                instance_id: SWEbenchHarnessInstanceResult(
                    instance_id=instance_id,
                    resolved=False,
                    patch_applied=False,
                    harness_status="missing_report",
                    evaluation_log_path=None,
                )
                for instance_id in instance_ids
            }
        normalized = self._normalize_report_payload(report_payload, output_dir=output_dir)
        return {
            instance_id: normalized.get(
                instance_id,
                SWEbenchHarnessInstanceResult(
                    instance_id=instance_id,
                    resolved=False,
                    patch_applied=False,
                    harness_status="not_reported",
                    evaluation_log_path=None,
                ),
            )
            for instance_id in instance_ids
        }

    def _load_report_payload(
        self,
        *,
        output_dir: Path,
        run_id: str,
    ) -> dict[str, Any] | list[Any] | None:
        candidates = [
            path
            for path in sorted(output_dir.glob("*.json"))
            if path.name != "swebench_harness_results.json"
        ]
        prioritized = sorted(
            candidates,
            key=lambda path: (
                run_id not in path.name,
                "predictions" not in path.name,
                path.name,
            ),
        )
        for candidate in prioritized:
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if self._looks_like_report_payload(payload):
                if isinstance(payload, (dict, list)):
                    return payload
        return None

    def _looks_like_report_payload(self, payload: Any) -> bool:
        if isinstance(payload, dict):
            keys = set(payload)
            if {"resolved_ids", "unresolved_ids"} & keys:
                return True
            if keys and all(isinstance(value, dict) for value in payload.values()):
                return True
        if isinstance(payload, list):
            return all(isinstance(item, dict) for item in payload)
        return False

    def _normalize_report_payload(
        self,
        payload: dict[str, Any] | list[Any],
        *,
        output_dir: Path,
    ) -> dict[str, SWEbenchHarnessInstanceResult]:
        if isinstance(payload, list):
            return {
                str(item["instance_id"]): SWEbenchHarnessInstanceResult(
                    instance_id=str(item["instance_id"]),
                    resolved=bool(item.get("resolved")),
                    patch_applied=bool(item.get("patch_applied")),
                    harness_status=str(item.get("harness_status", "completed")),
                    evaluation_log_path=self._coerce_log_path(
                        item.get("evaluation_log_path"),
                        output_dir,
                    ),
                )
                for item in payload
                if "instance_id" in item
            }
        if {"resolved_ids", "unresolved_ids"} & set(payload):
            resolved_ids = {str(item) for item in payload.get("resolved_ids", [])}
            unresolved_ids = {str(item) for item in payload.get("unresolved_ids", [])}
            applied_ids = {str(item) for item in payload.get("applied_ids", [])}
            results: dict[str, SWEbenchHarnessInstanceResult] = {}
            for instance_id in sorted(resolved_ids | unresolved_ids | applied_ids):
                resolved = instance_id in resolved_ids
                status = "resolved" if resolved else "unresolved"
                results[instance_id] = SWEbenchHarnessInstanceResult(
                    instance_id=instance_id,
                    resolved=resolved,
                    patch_applied=instance_id in applied_ids or resolved,
                    harness_status=status,
                    evaluation_log_path=self._guess_log_path(instance_id, output_dir),
                )
            return results
        results = {}
        for instance_id, value in payload.items():
            if not isinstance(value, dict):
                continue
            patch_applied = bool(
                value.get("patch_applied", value.get("patch_successfully_applied", False))
            )
            results[str(instance_id)] = SWEbenchHarnessInstanceResult(
                instance_id=str(instance_id),
                resolved=bool(value.get("resolved", False)),
                patch_applied=patch_applied,
                harness_status=str(value.get("status", value.get("harness_status", "completed"))),
                evaluation_log_path=self._coerce_log_path(
                    value.get("log_file", value.get("evaluation_log_path")),
                    output_dir,
                ),
            )
        return results

    def _guess_log_path(self, instance_id: str, output_dir: Path) -> str | None:
        candidates = sorted(output_dir.rglob(f"*{instance_id}*.log"))
        if not candidates:
            return None
        return str(candidates[0])

    def _coerce_log_path(self, value: object, output_dir: Path) -> str | None:
        if not isinstance(value, str) or not value:
            return None
        path = Path(value)
        if not path.is_absolute():
            path = (output_dir / path).resolve()
        return str(path)


__all__ = [
    "SWEbenchHarnessInstanceResult",
    "SWEbenchHarnessRunArtifacts",
    "SWEbenchHarnessRunner",
]
