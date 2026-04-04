from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from statistics import median
from typing import Any

from vtm.benchmarks.models import BenchmarkCaseResult, BenchmarkRunResult, BenchmarkSuite
from vtm.enums import ValidityStatus


class BenchmarkReporter:
    def summarize_results(
        self,
        suite: BenchmarkSuite,
        results: list[BenchmarkCaseResult],
    ) -> dict[str, Any]:
        if not results:
            return {}

        metrics = [result.metrics for result in results]
        if suite == "retrieval":
            summary = self._summarize_retrieval_metrics(results)
            summary["slice_metrics"] = self._summarize_retrieval_slices(results)
            return summary

        if suite == "drift":
            confusion: dict[str, int] = {}
            expected_relocated = 0
            predicted_relocated = 0
            correctly_relocated = 0
            non_verified_expected = 0
            false_verified = 0
            latencies = [float(metric["latency_ms"]) for metric in metrics]
            for metric in metrics:
                expected = str(metric["expected_status"])
                predicted = str(metric["predicted_status"])
                transition = f"{expected}->{predicted}"
                confusion[transition] = confusion.get(transition, 0) + 1
                if expected == ValidityStatus.RELOCATED.value:
                    expected_relocated += 1
                if predicted == ValidityStatus.RELOCATED.value:
                    predicted_relocated += 1
                if expected == predicted == ValidityStatus.RELOCATED.value:
                    correctly_relocated += 1
                if expected != ValidityStatus.VERIFIED.value:
                    non_verified_expected += 1
                    if predicted == ValidityStatus.VERIFIED.value:
                        false_verified += 1
            return {
                "status_confusion": confusion,
                "relocation_precision": 0.0
                if predicted_relocated == 0
                else correctly_relocated / predicted_relocated,
                "relocation_recall": 0.0
                if expected_relocated == 0
                else correctly_relocated / expected_relocated,
                "false_verified_rate": 0.0
                if non_verified_expected == 0
                else false_verified / non_verified_expected,
                "median_verification_latency_ms": median(latencies),
            }

        executed = [metric for metric in metrics if bool(metric["executed"])]
        testable = [metric for metric in metrics if bool(metric["testable"])]
        evaluated = [metric for metric in metrics if bool(metric.get("evaluated", False))]
        backends: dict[str, int] = {}
        for result in results:
            backend = str(result.metadata.get("evaluation_backend", "local_subprocess"))
            backends[backend] = backends.get(backend, 0) + 1
        runtimes = [float(metric["runtime_ms"]) for metric in executed]
        similarities = [
            float(metric["patch_similarity"])
            for metric in executed
            if metric["patch_similarity"] is not None
        ]
        changed_path_f1s = [
            float(metric["changed_path_f1"])
            for metric in executed
            if metric["changed_path_f1"] is not None
        ]
        return {
            "total_tasks": len(metrics),
            "executed_count": len(executed),
            "resolved_count": sum(1 for metric in metrics if bool(metric.get("resolved", False))),
            "testable_task_count": len(testable),
            "completed_testable_count": len(evaluated),
            "pass_rate": 0.0
            if not evaluated
            else self._mean(1.0 if bool(metric["passed"]) else 0.0 for metric in evaluated),
            "resolved_rate": 0.0
            if not evaluated
            else self._mean(
                1.0 if bool(metric.get("resolved", False)) else 0.0
                for metric in evaluated
            ),
            "patch_applied_rate": 0.0
            if not evaluated
            else self._mean(
                1.0 if bool(metric.get("patch_applied", False)) else 0.0 for metric in evaluated
            ),
            "median_runtime_ms": 0.0 if not runtimes else median(runtimes),
            "median_patch_similarity": 0.0 if not similarities else median(similarities),
            "mean_changed_path_f1": 0.0
            if not changed_path_f1s
            else self._mean(changed_path_f1s),
            "retrieval_usage_rate": self._mean(
                float(metric["retrieval_usage_rate"]) for metric in metrics
            ),
            "median_context_chars": median(int(metric["context_chars"]) for metric in metrics),
            "evaluation_backend_breakdown": backends,
        }

    def _summarize_retrieval_metrics(
        self,
        results: list[BenchmarkCaseResult],
    ) -> dict[str, Any]:
        metrics = [result.metrics for result in results]
        latencies = [float(metric["latency_ms"]) for metric in metrics]
        return {
            "recall_at_1": self._mean(float(metric["recall_at_1"]) for metric in metrics),
            "recall_at_3": self._mean(float(metric["recall_at_3"]) for metric in metrics),
            "recall_at_5": self._mean(float(metric["recall_at_5"]) for metric in metrics),
            "mrr": self._mean(float(metric["mrr"]) for metric in metrics),
            "ndcg": self._mean(float(metric["ndcg"]) for metric in metrics),
            "median_latency_ms": median(latencies),
            "artifact_bytes_per_memory": self._mean(
                float(metric["artifact_bytes_per_memory"]) for metric in metrics
            ),
        }

    def _summarize_retrieval_slices(
        self,
        results: list[BenchmarkCaseResult],
    ) -> dict[str, Any]:
        slices: dict[str, list[BenchmarkCaseResult]] = {}
        for result in results:
            slice_name = str(result.metadata.get("slice_name", "default"))
            slices.setdefault(slice_name, []).append(result)
        return {
            slice_name: {
                "case_count": len(slice_results),
                **self._summarize_retrieval_metrics(slice_results),
            }
            for slice_name, slice_results in sorted(slices.items())
        }

    def render_summary(self, result: BenchmarkRunResult) -> str:
        lines = [
            "# VTM Benchmark Summary",
            "",
            f"- Run ID: `{result.run_id}`",
            f"- Manifest: `{result.manifest_id}`",
            f"- Suite: `{result.suite}`",
            f"- Mode: `{result.mode}`",
            f"- Cases: `{result.case_count}`",
            "",
            "## Metrics",
        ]
        for key, value in sorted(result.metrics.items()):
            lines.append(f"- {key}: `{value}`")
        return "\n".join(lines) + "\n"

    def write_jsonl(self, path: Path, rows: list[dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True))
                handle.write("\n")

    def _mean(self, values: Iterable[float]) -> float:
        collected = list(values)
        if not collected:
            return 0.0
        return sum(collected) / len(collected)
