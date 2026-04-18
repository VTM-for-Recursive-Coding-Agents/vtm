"""Summary and report rendering helpers for benchmark runs."""

from __future__ import annotations

import json
import math
import random
from collections.abc import Iterable
from pathlib import Path
from statistics import median
from typing import Any

from vtm.benchmarks.models import (
    BenchmarkAttemptResult,
    BenchmarkCaseResult,
    BenchmarkComparisonResult,
    BenchmarkRunResult,
    BenchmarkSuite,
)
from vtm.enums import ValidityStatus

VALID_RETRIEVAL_STATUSES = frozenset(
    (ValidityStatus.VERIFIED.value, ValidityStatus.RELOCATED.value)
)


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
        workspace_backends: dict[str, int] = {}
        for result in results:
            backend = str(result.metadata.get("evaluation_backend", "local_subprocess"))
            backends[backend] = backends.get(backend, 0) + 1
            workspace_backend = str(
                result.metadata.get("workspace_backend", "local_workspace")
            )
            workspace_backends[workspace_backend] = (
                workspace_backends.get(workspace_backend, 0) + 1
            )
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
        verified_counts = [int(metric.get("verified_count", 0)) for metric in metrics]
        relocated_counts = [int(metric.get("relocated_count", 0)) for metric in metrics]
        stale_filtered_counts = [int(metric.get("stale_filtered_count", 0)) for metric in metrics]
        stale_hit_rates = [float(metric.get("stale_hit_rate", 0.0)) for metric in metrics]
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
            "mean_verified_count": self._mean(verified_counts),
            "mean_relocated_count": self._mean(relocated_counts),
            "mean_stale_filtered_count": self._mean(stale_filtered_counts),
            "mean_stale_hit_rate": self._mean(stale_hit_rates),
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
            "workspace_backend_breakdown": workspace_backends,
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
            summary["execution_style_metrics"] = self._summarize_coding_breakdown(
                results,
                attempts_by_case,
                key_name="execution_style",
                pass_k_values=requested_pass_k,
            )
            summary["workspace_backend_metrics"] = self._summarize_coding_breakdown(
                results,
                attempts_by_case,
                key_name="workspace_backend",
                pass_k_values=requested_pass_k,
            )
        return summary

    def _summarize_retrieval_metrics(
        self,
        results: list[BenchmarkCaseResult],
    ) -> dict[str, Any]:
        metrics = [result.metrics for result in results]
        latencies = [float(metric["latency_ms"]) for metric in metrics]
        summary = {
            "recall_at_1": self._mean(float(metric["recall_at_1"]) for metric in metrics),
            "recall_at_3": self._mean(float(metric["recall_at_3"]) for metric in metrics),
            "recall_at_5": self._mean(float(metric["recall_at_5"]) for metric in metrics),
            "mrr": self._mean(float(metric["mrr"]) for metric in metrics),
            "ndcg": self._mean(float(metric["ndcg"]) for metric in metrics),
            "median_latency_ms": median(latencies),
            "artifact_bytes_per_memory": self._mean(
                float(metric["artifact_bytes_per_memory"]) for metric in metrics
            ),
            "mean_verified_count": self._mean(
                float(metric.get("verified_count", 0.0)) for metric in metrics
            ),
            "mean_relocated_count": self._mean(
                float(metric.get("relocated_count", 0.0)) for metric in metrics
            ),
            "mean_stale_filtered_count": self._mean(
                float(metric.get("stale_filtered_count", 0.0)) for metric in metrics
            ),
            "mean_stale_hit_rate": self._mean(
                float(metric.get("stale_hit_rate", 0.0)) for metric in metrics
            ),
        }
        drifted_results = [
            result
            for result in results
            if result.metadata.get("expected_head_status") is not None
        ]
        if not drifted_results:
            return summary

        valid_results = [
            result
            for result in drifted_results
            if str(result.metadata.get("expected_head_status")) in VALID_RETRIEVAL_STATUSES
        ]
        stale_results = [
            result
            for result in drifted_results
            if str(result.metadata.get("expected_head_status")) == ValidityStatus.STALE.value
        ]
        summary["valid_recall_at_1"] = self._mean(
            float(result.metrics["recall_at_1"]) for result in valid_results
        )
        summary["valid_recall_at_3"] = self._mean(
            float(result.metrics["recall_at_3"]) for result in valid_results
        )
        summary["valid_recall_at_5"] = self._mean(
            float(result.metrics["recall_at_5"]) for result in valid_results
        )
        summary["stale_rejection_rate"] = self._mean(
            1.0 if result.metrics.get("rank") is None else 0.0 for result in stale_results
        )
        summary["stale_hit_rate"] = self._mean(
            1.0 if result.metrics.get("rank") is not None else 0.0 for result in stale_results
        )
        summary["safe_retrieval_at_1"] = self._mean(
            self._safe_retrieval_hit(result, k=1) for result in drifted_results
        )
        summary["safe_retrieval_at_5"] = self._mean(
            self._safe_retrieval_hit(result, k=5) for result in drifted_results
        )
        return summary

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

    def compare_runs(
        self,
        *,
        baseline: BenchmarkRunResult,
        candidate: BenchmarkRunResult,
        baseline_results: list[BenchmarkCaseResult],
        candidate_results: list[BenchmarkCaseResult],
        baseline_attempts: list[BenchmarkAttemptResult] | None = None,
        candidate_attempts: list[BenchmarkAttemptResult] | None = None,
        bootstrap_samples: int = 2000,
        bootstrap_seed: int = 0,
    ) -> BenchmarkComparisonResult:
        """Build a paired comparison result for two completed benchmark runs."""
        if baseline.suite != candidate.suite:
            raise ValueError(
                "benchmark comparisons require matching suites: "
                f"{baseline.suite} != {candidate.suite}"
            )

        baseline_by_case = {result.case_id: result for result in baseline_results}
        candidate_by_case = {result.case_id: result for result in candidate_results}
        common_case_ids = sorted(set(baseline_by_case) & set(candidate_by_case))
        if not common_case_ids:
            raise ValueError("benchmark comparisons require at least one shared case_id")

        metrics: dict[str, Any] = {
            "baseline_summary_metrics": baseline.metrics,
            "candidate_summary_metrics": candidate.metrics,
            "summary_scalar_deltas": self._compare_summary_scalars(
                baseline.metrics,
                candidate.metrics,
            ),
            "paired_numeric_metrics": self._compare_case_numeric_metrics(
                common_case_ids,
                baseline_by_case,
                candidate_by_case,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed,
            ),
            "paired_binary_metrics": self._compare_case_binary_metrics(
                common_case_ids,
                baseline_by_case,
                candidate_by_case,
            ),
        }

        if baseline.suite == "coding":
            metrics["paired_attempt_binary_metrics"] = self._compare_attempt_success_metrics(
                common_case_ids,
                baseline.metrics,
                candidate.metrics,
                self._group_attempts_by_case(baseline_attempts or []),
                self._group_attempts_by_case(candidate_attempts or []),
            )

        comparison_id = f"{baseline.run_id}-vs-{candidate.run_id}"
        return BenchmarkComparisonResult(
            comparison_id=comparison_id,
            suite=baseline.suite,
            baseline_run_id=baseline.run_id,
            baseline_manifest_id=baseline.manifest_id,
            baseline_mode=baseline.mode,
            baseline_case_count=baseline.case_count,
            candidate_run_id=candidate.run_id,
            candidate_manifest_id=candidate.manifest_id,
            candidate_mode=candidate.mode,
            candidate_case_count=candidate.case_count,
            common_case_count=len(common_case_ids),
            baseline_only_case_count=len(set(baseline_by_case) - set(candidate_by_case)),
            candidate_only_case_count=len(set(candidate_by_case) - set(baseline_by_case)),
            metrics=metrics,
        )

    def render_comparison(self, comparison: BenchmarkComparisonResult) -> str:
        """Render a human-readable Markdown summary for a benchmark comparison."""
        lines = [
            "# VTM Benchmark Comparison",
            "",
            f"- Suite: `{comparison.suite}`",
            (
                f"- Baseline: `{comparison.baseline_run_id}` "
                f"(`{comparison.baseline_mode}`, manifest `{comparison.baseline_manifest_id}`)"
            ),
            (
                f"- Candidate: `{comparison.candidate_run_id}` "
                f"(`{comparison.candidate_mode}`, manifest `{comparison.candidate_manifest_id}`)"
            ),
            f"- Common cases: `{comparison.common_case_count}`",
            f"- Baseline-only cases: `{comparison.baseline_only_case_count}`",
            f"- Candidate-only cases: `{comparison.candidate_only_case_count}`",
            "",
            "## Summary Scalar Deltas",
        ]
        summary_scalars = comparison.metrics.get("summary_scalar_deltas", {})
        for key, value in sorted(summary_scalars.items()):
            lines.append(
                "- "
                f"{key}: `{value['baseline_value']} -> {value['candidate_value']} "
                f"(delta {value['delta']})`"
            )
        if not summary_scalars:
            lines.append("- none")

        lines.extend(["", "## Paired Binary Metrics"])
        paired_binary = comparison.metrics.get("paired_binary_metrics", {})
        for key, value in sorted(paired_binary.items()):
            lines.append(
                "- "
                f"{key}: `baseline_rate={value['baseline_rate']} "
                f"candidate_rate={value['candidate_rate']} "
                f"delta={value['rate_delta']} "
                f"candidate_only_true={value['candidate_only_true_count']} "
                f"baseline_only_true={value['baseline_only_true_count']} "
                f"p={value['mcnemar_p_value']}`"
            )
        if not paired_binary:
            lines.append("- none")

        paired_attempt = comparison.metrics.get("paired_attempt_binary_metrics", {})
        if paired_attempt:
            lines.extend(["", "## Paired Attempt Metrics"])
            for key, value in sorted(paired_attempt.items()):
                lines.append(
                    "- "
                    f"{key}: `baseline_rate={value['baseline_rate']} "
                    f"candidate_rate={value['candidate_rate']} "
                    f"delta={value['rate_delta']} "
                    f"candidate_only_true={value['candidate_only_true_count']} "
                    f"baseline_only_true={value['baseline_only_true_count']} "
                    f"p={value['mcnemar_p_value']}`"
                )

        lines.extend(["", "## Paired Numeric Metrics"])
        paired_numeric = comparison.metrics.get("paired_numeric_metrics", {})
        for key, value in sorted(paired_numeric.items()):
            lines.append(
                "- "
                f"{key}: `baseline_mean={value['baseline_mean']} "
                f"candidate_mean={value['candidate_mean']} "
                f"mean_delta={value['mean_delta']} "
                f"ci95={tuple(value['bootstrap_ci_95'])}`"
            )
        if not paired_numeric:
            lines.append("- none")
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

    def _safe_retrieval_hit(self, result: BenchmarkCaseResult, *, k: int) -> float:
        expected_status = str(result.metadata.get("expected_head_status"))
        if expected_status in VALID_RETRIEVAL_STATUSES:
            metric_name = f"recall_at_{k}"
            return 1.0 if float(result.metrics.get(metric_name, 0.0)) == 1.0 else 0.0
        if expected_status == ValidityStatus.STALE.value:
            return 1.0 if result.metrics.get("rank") is None else 0.0
        return 0.0

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

    def _compare_summary_scalars(
        self,
        baseline_metrics: dict[str, Any],
        candidate_metrics: dict[str, Any],
    ) -> dict[str, Any]:
        summary: dict[str, Any] = {}
        for key in sorted(set(baseline_metrics) & set(candidate_metrics)):
            baseline_value = baseline_metrics[key]
            candidate_value = candidate_metrics[key]
            if not self._is_numeric_scalar(baseline_value) or not self._is_numeric_scalar(
                candidate_value
            ):
                continue
            baseline_float = float(baseline_value)
            candidate_float = float(candidate_value)
            summary[key] = {
                "baseline_value": baseline_float,
                "candidate_value": candidate_float,
                "delta": candidate_float - baseline_float,
            }
        return summary

    def _compare_case_numeric_metrics(
        self,
        common_case_ids: list[str],
        baseline_by_case: dict[str, BenchmarkCaseResult],
        candidate_by_case: dict[str, BenchmarkCaseResult],
        *,
        bootstrap_samples: int,
        bootstrap_seed: int,
    ) -> dict[str, Any]:
        paired_values: dict[str, list[tuple[float, float]]] = {}
        for case_id in common_case_ids:
            baseline_metrics = baseline_by_case[case_id].metrics
            candidate_metrics = candidate_by_case[case_id].metrics
            for key in set(baseline_metrics) & set(candidate_metrics):
                baseline_value = baseline_metrics[key]
                candidate_value = candidate_metrics[key]
                if not self._is_numeric_scalar(baseline_value) or not self._is_numeric_scalar(
                    candidate_value
                ):
                    continue
                paired_values.setdefault(str(key), []).append(
                    (float(baseline_value), float(candidate_value))
                )

        summary: dict[str, Any] = {}
        for index, key in enumerate(sorted(paired_values)):
            pairs = paired_values[key]
            baseline_values = [baseline_value for baseline_value, _ in pairs]
            candidate_values = [candidate_value for _, candidate_value in pairs]
            deltas = [candidate_value - baseline_value for baseline_value, candidate_value in pairs]
            epsilon = 1e-12
            summary[key] = {
                "paired_case_count": len(pairs),
                "baseline_mean": self._mean(baseline_values),
                "candidate_mean": self._mean(candidate_values),
                "mean_delta": self._mean(deltas),
                "median_delta": median(deltas),
                "candidate_greater_count": sum(1 for delta in deltas if delta > epsilon),
                "baseline_greater_count": sum(1 for delta in deltas if delta < -epsilon),
                "equal_count": sum(1 for delta in deltas if abs(delta) <= epsilon),
                "bootstrap_ci_95": self._bootstrap_mean_delta_ci(
                    deltas,
                    samples=bootstrap_samples,
                    seed=bootstrap_seed + index,
                ),
            }
        return summary

    def _compare_case_binary_metrics(
        self,
        common_case_ids: list[str],
        baseline_by_case: dict[str, BenchmarkCaseResult],
        candidate_by_case: dict[str, BenchmarkCaseResult],
    ) -> dict[str, Any]:
        paired_values: dict[str, list[tuple[bool, bool]]] = {}
        for case_id in common_case_ids:
            baseline_metrics = baseline_by_case[case_id].metrics
            candidate_metrics = candidate_by_case[case_id].metrics
            for key in set(baseline_metrics) & set(candidate_metrics):
                baseline_value = baseline_metrics[key]
                candidate_value = candidate_metrics[key]
                if not self._is_bool_scalar(baseline_value) or not self._is_bool_scalar(
                    candidate_value
                ):
                    continue
                paired_values.setdefault(str(key), []).append((baseline_value, candidate_value))

        return {
            key: self._paired_binary_metric_summary(pairs)
            for key, pairs in sorted(paired_values.items())
        }

    def _compare_attempt_success_metrics(
        self,
        common_case_ids: list[str],
        baseline_summary_metrics: dict[str, Any],
        candidate_summary_metrics: dict[str, Any],
        baseline_attempts_by_case: dict[str, list[BenchmarkAttemptResult]],
        candidate_attempts_by_case: dict[str, list[BenchmarkAttemptResult]],
    ) -> dict[str, Any]:
        if not baseline_attempts_by_case or not candidate_attempts_by_case:
            return {}

        summary: dict[str, Any] = {}
        for metric_name, summary_prefix in (
            ("passed", "pass_at_"),
            ("resolved", "resolved_at_"),
            ("patch_applied", "patch_applied_at_"),
        ):
            for k in sorted(
                self._shared_attempt_k_values(
                    baseline_summary_metrics,
                    candidate_summary_metrics,
                    prefix=summary_prefix,
                )
            ):
                pairs = [
                    (
                        self._attempt_metric_succeeded(
                            baseline_attempts_by_case.get(case_id, []),
                            k,
                            metric_name=metric_name,
                        ),
                        self._attempt_metric_succeeded(
                            candidate_attempts_by_case.get(case_id, []),
                            k,
                            metric_name=metric_name,
                        ),
                    )
                    for case_id in common_case_ids
                ]
                summary[f"{summary_prefix}{k}"] = self._paired_binary_metric_summary(pairs)
        return summary

    def _shared_attempt_k_values(
        self,
        baseline_summary_metrics: dict[str, Any],
        candidate_summary_metrics: dict[str, Any],
        *,
        prefix: str,
    ) -> set[int]:
        baseline_values = {
            int(key.removeprefix(prefix))
            for key in baseline_summary_metrics
            if key.startswith(prefix) and key.removeprefix(prefix).isdigit()
        }
        candidate_values = {
            int(key.removeprefix(prefix))
            for key in candidate_summary_metrics
            if key.startswith(prefix) and key.removeprefix(prefix).isdigit()
        }
        return baseline_values & candidate_values

    def _attempt_metric_succeeded(
        self,
        attempts: list[BenchmarkAttemptResult],
        k: int,
        *,
        metric_name: str,
    ) -> bool:
        return any(bool(attempt.metrics.get(metric_name, False)) for attempt in attempts[:k])

    def _paired_binary_metric_summary(self, pairs: list[tuple[bool, bool]]) -> dict[str, Any]:
        baseline_true = sum(1 for baseline, _ in pairs if baseline)
        candidate_true = sum(1 for _, candidate in pairs if candidate)
        candidate_only_true = sum(1 for baseline, candidate in pairs if not baseline and candidate)
        baseline_only_true = sum(1 for baseline, candidate in pairs if baseline and not candidate)
        both_true = sum(1 for baseline, candidate in pairs if baseline and candidate)
        both_false = sum(1 for baseline, candidate in pairs if not baseline and not candidate)
        paired_case_count = len(pairs)
        return {
            "paired_case_count": paired_case_count,
            "baseline_rate": 0.0 if paired_case_count == 0 else baseline_true / paired_case_count,
            "candidate_rate": 0.0
            if paired_case_count == 0
            else candidate_true / paired_case_count,
            "rate_delta": 0.0
            if paired_case_count == 0
            else (candidate_true - baseline_true) / paired_case_count,
            "candidate_only_true_count": candidate_only_true,
            "baseline_only_true_count": baseline_only_true,
            "both_true_count": both_true,
            "both_false_count": both_false,
            "mcnemar_p_value": self._mcnemar_exact_p_value(
                baseline_only_true,
                candidate_only_true,
            ),
        }

    def _bootstrap_mean_delta_ci(
        self,
        deltas: list[float],
        *,
        samples: int,
        seed: int,
    ) -> list[float]:
        if not deltas:
            return [0.0, 0.0]
        if len(deltas) == 1:
            return [deltas[0], deltas[0]]
        rng = random.Random(seed)
        size = len(deltas)
        bootstrap_means = []
        for _ in range(samples):
            sample = [deltas[rng.randrange(size)] for _ in range(size)]
            bootstrap_means.append(sum(sample) / size)
        bootstrap_means.sort()
        return [
            self._percentile(bootstrap_means, 0.025),
            self._percentile(bootstrap_means, 0.975),
        ]

    def _percentile(self, values: list[float], percentile: float) -> float:
        if not values:
            return 0.0
        if len(values) == 1:
            return values[0]
        position = percentile * (len(values) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return values[lower]
        weight = position - lower
        return values[lower] * (1.0 - weight) + values[upper] * weight

    def _mcnemar_exact_p_value(
        self,
        baseline_only_true_count: int,
        candidate_only_true_count: int,
    ) -> float:
        discordant = baseline_only_true_count + candidate_only_true_count
        if discordant == 0:
            return 1.0
        lower_tail = sum(
            math.comb(discordant, value)
            for value in range(0, min(baseline_only_true_count, candidate_only_true_count) + 1)
        ) / (2**discordant)
        return float(min(1.0, 2.0 * lower_tail))

    def _is_numeric_scalar(self, value: Any) -> bool:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        return math.isfinite(float(value))

    def _is_bool_scalar(self, value: Any) -> bool:
        return isinstance(value, bool)
