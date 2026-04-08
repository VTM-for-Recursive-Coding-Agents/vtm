"""Summary and report rendering helpers for benchmark runs."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from statistics import median
from typing import Any

from vtm.benchmarks.models import (
    BenchmarkAttemptResult,
    BenchmarkCaseResult,
    BenchmarkRunResult,
    BenchmarkSuite,
)
from vtm.enums import ValidityStatus


class BenchmarkReporter:
    """Aggregates per-case metrics into run-level summaries and artifacts."""

    def summarize_results(
        self,
        suite: BenchmarkSuite,
        results: list[BenchmarkCaseResult],
        *,
        attempts: list[BenchmarkAttemptResult] | None = None,
        pass_k_values: tuple[int, ...] = (1,),
    ) -> dict[str, Any]:
        """Summarize per-case results for the requested suite."""
        if not results:
            return {}

        metrics = [result.metrics for result in results]
        coding_attempts = attempts or []
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

        attempts_by_case = self._group_attempts_by_case(coding_attempts)
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
        turn_counts = [
            int(metric["turn_count"]) for metric in metrics if metric.get("turn_count") is not None
        ]
        tool_call_counts = [
            int(metric["tool_call_count"])
            for metric in metrics
            if metric.get("tool_call_count") is not None
        ]
        tool_failure_counts = [
            int(metric["tool_failure_count"])
            for metric in metrics
            if metric.get("tool_failure_count") is not None
        ]
        compaction_counts = [
            int(metric["compaction_count"])
            for metric in metrics
            if metric.get("compaction_count") is not None
        ]
        test_iterations = [
            int(metric["test_iterations"])
            for metric in metrics
            if metric.get("test_iterations") is not None
        ]
        memory_write_counts = [
            int(metric["memory_write_count"])
            for metric in metrics
            if metric.get("memory_write_count") is not None
        ]
        memory_promotion_counts = [
            int(metric["memory_promotion_count"])
            for metric in metrics
            if metric.get("memory_promotion_count") is not None
        ]
        guardrail_blocks = [
            int(metric["guardrail_blocks"])
            for metric in metrics
            if metric.get("guardrail_blocks") is not None
        ]
        terminal_command_counts = [
            int(metric["terminal_command_count"])
            for metric in metrics
            if metric.get("terminal_command_count") is not None
        ]
        command_timeout_counts = [
            int(metric["command_timeout_count"])
            for metric in metrics
            if metric.get("command_timeout_count") is not None
        ]
        attempt_runtimes = [
            float(attempt.metrics["runtime_ms"])
            for attempt in coding_attempts
            if attempt.metrics.get("executed")
        ]
        final_verification_metrics = [
            metric
            for metric in metrics
            if metric.get("final_verification_passed") is not None
        ]
        agent_statuses = [
            str(result.metadata.get("agent_status"))
            for result in results
            if result.metadata.get("agent_status") is not None
        ]
        failure_breakdown: dict[str, int] = {}
        for result in results:
            category = self._classify_coding_failure(result)
            if category is None:
                continue
            failure_breakdown[category] = failure_breakdown.get(category, 0) + 1
        agent_status_breakdown: dict[str, int] = {}
        for status in agent_statuses:
            agent_status_breakdown[status] = agent_status_breakdown.get(status, 0) + 1
        summary = {
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
            "median_turn_count": 0.0 if not turn_counts else median(turn_counts),
            "mean_tool_call_count": 0.0 if not tool_call_counts else self._mean(tool_call_counts),
            "mean_tool_failure_count": 0.0
            if not tool_failure_counts
            else self._mean(tool_failure_counts),
            "mean_compaction_count": 0.0
            if not compaction_counts
            else self._mean(compaction_counts),
            "mean_test_iterations": 0.0
            if not test_iterations
            else self._mean(test_iterations),
            "mean_memory_write_count": 0.0
            if not memory_write_counts
            else self._mean(memory_write_counts),
            "mean_memory_promotion_count": 0.0
            if not memory_promotion_counts
            else self._mean(memory_promotion_counts),
            "mean_guardrail_blocks": 0.0
            if not guardrail_blocks
            else self._mean(guardrail_blocks),
            "mean_terminal_command_count": 0.0
            if not terminal_command_counts
            else self._mean(terminal_command_counts),
            "mean_command_timeout_count": 0.0
            if not command_timeout_counts
            else self._mean(command_timeout_counts),
            "final_verification_pass_rate": 0.0
            if not final_verification_metrics
            else self._mean(
                1.0 if bool(metric["final_verification_passed"]) else 0.0
                for metric in final_verification_metrics
            ),
            "infra_failure_rate": self._mean(
                1.0 if bool(metric.get("infra_failure", False)) else 0.0 for metric in metrics
            ),
            "agent_status_breakdown": agent_status_breakdown,
            "failure_breakdown": failure_breakdown,
            "evaluation_backend_breakdown": backends,
        }
        if suite == "coding":
            requested_pass_k = tuple(sorted(set(pass_k_values)))
            for k in requested_pass_k:
                summary[f"pass_at_{k}"] = self._attempt_success_rate(
                    attempts_by_case,
                    k,
                    metric_name="passed",
                )
                summary[f"resolved_at_{k}"] = self._attempt_success_rate(
                    attempts_by_case,
                    k,
                    metric_name="resolved",
                )
                summary[f"patch_applied_at_{k}"] = self._attempt_success_rate(
                    attempts_by_case,
                    k,
                    metric_name="patch_applied",
                )
            summary["median_attempt_runtime_ms"] = (
                0.0 if not attempt_runtimes else median(attempt_runtimes)
            )
            summary["mean_attempts_executed"] = self._mean(
                float(
                    sum(
                        1
                        for attempt in attempts_by_case.get(result.case_id, [])
                        if bool(attempt.metrics.get("executed", False))
                    )
                )
                for result in results
            )
            best_attempt_indices = [
                int(result.metadata["best_attempt_index"])
                for result in results
                if result.metadata.get("best_attempt_index") is not None
            ]
            summary["mean_best_attempt_index"] = (
                0.0 if not best_attempt_indices else self._mean(best_attempt_indices)
            )
            summary["difficulty_metrics"] = self._summarize_coding_breakdown(
                results,
                attempts_by_case,
                key_name="difficulty",
                pass_k_values=requested_pass_k,
            )
            summary["task_kind_metrics"] = self._summarize_coding_breakdown(
                results,
                attempts_by_case,
                key_name="task_kind",
                pass_k_values=requested_pass_k,
            )
            summary["evaluation_backend_metrics"] = self._summarize_coding_breakdown(
                results,
                attempts_by_case,
                key_name="evaluation_backend",
                pass_k_values=requested_pass_k,
            )
        return summary

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
        """Render a human-readable Markdown summary for a benchmark run."""
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

    def _group_attempts_by_case(
        self,
        attempts: list[BenchmarkAttemptResult],
    ) -> dict[str, list[BenchmarkAttemptResult]]:
        grouped: dict[str, list[BenchmarkAttemptResult]] = {}
        for attempt in attempts:
            grouped.setdefault(attempt.case_id, []).append(attempt)
        for case_attempts in grouped.values():
            case_attempts.sort(key=lambda item: item.attempt_index)
        return grouped

    def _attempt_success_rate(
        self,
        attempts_by_case: dict[str, list[BenchmarkAttemptResult]],
        k: int,
        *,
        metric_name: str,
    ) -> float:
        if not attempts_by_case:
            return 0.0
        successes = 0
        for attempts in attempts_by_case.values():
            if any(bool(attempt.metrics.get(metric_name, False)) for attempt in attempts[:k]):
                successes += 1
        return successes / len(attempts_by_case)

    def _summarize_coding_breakdown(
        self,
        results: list[BenchmarkCaseResult],
        attempts_by_case: dict[str, list[BenchmarkAttemptResult]],
        *,
        key_name: str,
        pass_k_values: tuple[int, ...],
    ) -> dict[str, Any]:
        grouped_results: dict[str, list[BenchmarkCaseResult]] = {}
        for result in results:
            key = str(result.metadata.get(key_name, "unknown"))
            grouped_results.setdefault(key, []).append(result)
        breakdown: dict[str, Any] = {}
        for key, grouped in sorted(grouped_results.items()):
            case_ids = {result.case_id for result in grouped}
            grouped_attempts = {
                case_id: attempts_by_case.get(case_id, [])
                for case_id in case_ids
            }
            item_summary: dict[str, Any] = {
                "case_count": len(grouped),
                "pass_rate": self._mean(
                    1.0 if bool(result.metrics.get("passed", False)) else 0.0
                    for result in grouped
                ),
                "resolved_rate": self._mean(
                    1.0 if bool(result.metrics.get("resolved", False)) else 0.0
                    for result in grouped
                ),
            }
            for k in pass_k_values:
                item_summary[f"pass_at_{k}"] = self._attempt_success_rate(
                    grouped_attempts,
                    k,
                    metric_name="passed",
                )
                item_summary[f"resolved_at_{k}"] = self._attempt_success_rate(
                    grouped_attempts,
                    k,
                    metric_name="resolved",
                )
                item_summary[f"patch_applied_at_{k}"] = self._attempt_success_rate(
                    grouped_attempts,
                    k,
                    metric_name="patch_applied",
                )
            breakdown[key] = item_summary
        return breakdown

    def _classify_coding_failure(self, result: BenchmarkCaseResult) -> str | None:
        metrics = result.metrics
        metadata = result.metadata
        if bool(metrics.get("infra_failure", False)):
            return "infra"
        if bool(metrics.get("passed", False)) or bool(metrics.get("resolved", False)):
            return None
        if metadata.get("agent_status") == "model_error":
            return "model"
        if metadata.get("agent_status") == "timeout" or bool(
            metrics.get("command_timeout_count", 0)
        ):
            return "timeout"
        if metadata.get("agent_status") == "max_tool_failures" or int(
            metrics.get("tool_failure_count", 0)
        ) > 0:
            return "tool"
        if bool(metrics.get("evaluated", False)):
            return "verification"
        return "infra"
