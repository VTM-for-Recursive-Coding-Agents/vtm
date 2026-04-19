#!/usr/bin/env python3
"""Build pass/fail and progress graphs from stored LiveCodeBench artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PassFailSummary:
    # Aggregated metrics for a single pass/fail export file.
    method: str
    file_path: Path
    total: int
    passed: int
    failed: int
    pass_rate: float
    fail_rate: float
    passed_ids: set[str]
    outcome_by_id: dict[str, str]


@dataclass
class IntersectionMethodMetric:
    method: str
    source_file: Path
    total_labeled: int
    passed_on_intersection: int
    failed_on_intersection: int
    pass_rate_on_intersection: float


@dataclass
class IntersectionSummary:
    model_prefix: str
    question_ids: set[str]
    methods: list[IntersectionMethodMetric]


@dataclass
class ChunkHealth:
    # Health metrics for one rlmfix chunk checkpoint file.
    kind: str
    chunk: str
    total: int
    nonempty: int
    nonempty_rate: float


def _method_from_filename(path: Path) -> str:
    # Normalize method labels so chart legends stay readable across naming variants.
    name = path.stem
    if name.endswith("_passfail"):
        name = name[: -len("_passfail")]
    prefixes = (
        "Qwen2.5-Coder-Ins-32B-",
        "Qwen2.5-Coder-32B-",
    )
    for prefix in prefixes:
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def _read_passfail(path: Path) -> PassFailSummary:
    passed_ids: set[str] = set()
    outcome_by_id: dict[str, str] = {}
    total = 0
    passed = 0
    failed = 0

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            total += 1
            question_id = str(row.get("question_id", "")).strip()
            status = str(row.get("pass_fail", "")).strip().upper()
            if status == "PASS":
                passed += 1
                if question_id:
                    passed_ids.add(question_id)
                    outcome_by_id[question_id] = status
            elif status == "FAIL":
                failed += 1
                if question_id:
                    outcome_by_id[question_id] = status

    fail_rate = (failed / total) if total else 0.0
    pass_rate = (passed / total) if total else 0.0
    return PassFailSummary(
        method=_method_from_filename(path),
        file_path=path,
        total=total,
        passed=passed,
        failed=failed,
        pass_rate=pass_rate,
        fail_rate=fail_rate,
        passed_ids=passed_ids,
        outcome_by_id=outcome_by_id,
    )


def _collect_passfail_summaries(runs_dir: Path) -> list[PassFailSummary]:
    # Collect all available method exports so the dashboard auto-updates as new files land.
    candidates = sorted(runs_dir.glob("*passfail.tsv"))
    summaries: list[PassFailSummary] = []
    for path in candidates:
        summaries.append(_read_passfail(path))
    return summaries


def _read_chunk_health(raw_dir: Path) -> list[ChunkHealth]:
    records: list[ChunkHealth] = []
    for progress_path in sorted(raw_dir.glob("*rlmfix*_a/rlm_progress.jsonl")):
        # Derive method kind and chunk id from the canonical run folder naming pattern.
        run_name = progress_path.parent.name
        kind = "rlm_rag" if "_rlm_rag_" in run_name else "rlm"
        chunk = "unknown"
        if "_rlmfix_" in run_name and "_d_a" in run_name:
            chunk = run_name.split("_rlmfix_", 1)[1].split("_d_a", 1)[0]

        total = 0
        nonempty = 0
        with progress_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                total += 1
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    # Keep processing even if a checkpoint file has occasional malformed lines.
                    continue
                outputs = payload.get("outputs")
                if isinstance(outputs, list):
                    text = "\n".join(item for item in outputs if isinstance(item, str))
                elif isinstance(outputs, str):
                    text = outputs
                else:
                    text = ""
                if text.strip():
                    nonempty += 1

        rate = (nonempty / total) if total else 0.0
        records.append(
            ChunkHealth(
                kind=kind,
                chunk=chunk,
                total=total,
                nonempty=nonempty,
                nonempty_rate=rate,
            )
        )
    return records


def _plot_pass_rate(ax, summaries: list[PassFailSummary]) -> None:
    # Topline quality chart: pass-rate percentage per method.
    labels = [s.method for s in summaries]
    values = [100.0 * s.pass_rate for s in summaries]
    bars = ax.bar(labels, values, color="#1f77b4")
    ax.set_title("Pass Rate By Method")
    ax.set_ylabel("Pass rate (%)")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=20)
    for bar, summary in zip(bars, summaries):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"n={summary.total}",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def _plot_stacked_counts(ax, summaries: list[PassFailSummary]) -> None:
    # Complements pass-rate by exposing sample-size differences across methods.
    labels = [s.method for s in summaries]
    pass_counts = [s.passed for s in summaries]
    fail_counts = [s.failed for s in summaries]
    ax.bar(labels, pass_counts, color="#2ca02c", label="PASS")
    ax.bar(labels, fail_counts, bottom=pass_counts, color="#d62728", label="FAIL")
    ax.set_title("Pass/Fail Counts By Method")
    ax.set_ylabel("Question count")
    ax.tick_params(axis="x", rotation=20)
    ax.legend()


def _plot_overlap(ax, summaries: list[PassFailSummary]) -> None:
    labels = [s.method for s in summaries]
    matrix: list[list[float]] = []
    for lhs in summaries:
        row: list[float] = []
        for rhs in summaries:
            # Jaccard overlap highlights whether methods solve the same or different questions.
            union = lhs.passed_ids | rhs.passed_ids
            inter = lhs.passed_ids & rhs.passed_ids
            row.append((len(inter) / len(union)) if union else 0.0)
        matrix.append(row)

    image = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="Blues")
    ax.set_title("Solved-Set Overlap (Jaccard)")
    ax.set_xticks(range(len(labels)), labels=labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    for i, row in enumerate(matrix):
        for j, val in enumerate(row):
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)
    return image


def _plot_chunk_health(ax, chunk_rows: list[ChunkHealth]) -> None:
    if not chunk_rows:
        ax.set_title("RLMFix Chunk Non-Empty Output Rate")
        ax.text(0.5, 0.5, "No rlm_progress.jsonl files found", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        return

    rows = sorted(chunk_rows, key=lambda r: (r.kind, r.chunk))
    # Color split helps visually separate rlm vs rlm_rag chunks.
    labels = [f"{r.kind}:{r.chunk}" for r in rows]
    values = [100.0 * r.nonempty_rate for r in rows]
    colors = ["#9467bd" if r.kind == "rlm_rag" else "#17becf" for r in rows]
    bars = ax.bar(labels, values, color=colors)
    ax.set_title("RLMFix Chunk Non-Empty Output Rate")
    ax.set_ylabel("Non-empty outputs (%)")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=55)
    for bar, row in zip(bars, rows):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{row.nonempty}/{row.total}",
            ha="center",
            va="bottom",
            fontsize=7,
        )


def _canonical_method_key(raw_method: str) -> tuple[str | None, bool, str]:
    normalized = raw_method.strip().lower().replace("-", "_")
    is_partial = "partial" in normalized
    if normalized.startswith("rlm_rag"):
        return "rlm_rag", is_partial, "rlm+rag"
    if normalized.startswith("baseline"):
        return "baseline", is_partial, "baseline"
    if normalized.startswith("rag"):
        return "rag", is_partial, "rag"
    if normalized.startswith("rlm"):
        return "rlm", is_partial, "rlm"
    return None, is_partial, raw_method


def _select_intersection_sources(
    summaries: list[PassFailSummary],
    model_prefix: str,
    partial_policy: str,
) -> dict[str, PassFailSummary]:
    required = ("baseline", "rag", "rlm", "rlm_rag")
    by_method: dict[str, list[tuple[bool, int, PassFailSummary]]] = {k: [] for k in required}
    for summary in summaries:
        name = summary.file_path.stem
        if name.endswith("_passfail"):
            name = name[: -len("_passfail")]
        if not name.startswith(model_prefix):
            continue
        method_label = _method_from_filename(summary.file_path)
        method_key, is_partial, _ = _canonical_method_key(method_label)
        if method_key in by_method:
            by_method[method_key].append((is_partial, summary.total, summary))

    selected: dict[str, PassFailSummary] = {}
    missing: list[str] = []
    for method_key in required:
        candidates = by_method.get(method_key, [])
        if not candidates:
            missing.append(method_key)
            continue
        # Policy-driven source selection:
        # - prefer_non_partial: prefer non-partial exports, then largest sample size.
        # - prefer_partial: prefer partial exports, then largest sample size.
        if partial_policy == "prefer_partial":
            candidates.sort(key=lambda item: (not item[0], -item[1]))
        else:
            candidates.sort(key=lambda item: (item[0], -item[1]))
        selected[method_key] = candidates[0][2]

    if missing:
        missing_text = ", ".join(missing)
        raise SystemExit(
            f"Intersection mode requires methods baseline/rag/rlm/rlm_rag for model prefix '{model_prefix}'. "
            f"Missing: {missing_text}."
        )
    return selected


def _compute_intersection_summary(
    summaries: list[PassFailSummary],
    model_prefix: str,
    partial_policy: str,
) -> IntersectionSummary:
    selected = _select_intersection_sources(summaries, model_prefix, partial_policy)
    question_id_sets = [set(s.outcome_by_id.keys()) for s in selected.values()]
    intersection_ids = set.intersection(*question_id_sets) if question_id_sets else set()

    metrics: list[IntersectionMethodMetric] = []
    for key in ("baseline", "rag", "rlm", "rlm_rag"):
        summary = selected[key]
        labels = summary.outcome_by_id
        passed = 0
        failed = 0
        for question_id in intersection_ids:
            if labels.get(question_id) == "PASS":
                passed += 1
            else:
                failed += 1
        total = len(intersection_ids)
        rate = (passed / total) if total else 0.0
        _, _, display_name = _canonical_method_key(key)
        metrics.append(
            IntersectionMethodMetric(
                method=display_name,
                source_file=summary.file_path,
                total_labeled=len(labels),
                passed_on_intersection=passed,
                failed_on_intersection=failed,
                pass_rate_on_intersection=rate,
            )
        )

    return IntersectionSummary(
        model_prefix=model_prefix,
        question_ids=intersection_ids,
        methods=metrics,
    )


def _plot_intersection_pass_rate(ax, intersection: IntersectionSummary) -> None:
    labels = [m.method for m in intersection.methods]
    values = [100.0 * m.pass_rate_on_intersection for m in intersection.methods]
    bars = ax.bar(labels, values, color="#4c78a8")
    denom = len(intersection.question_ids)
    ax.set_title("Intersection Pass@1 Rate By Method")
    ax.set_ylabel("Pass@1 (%)")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=20)
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"N={denom}",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def _plot_intersection_counts(ax, intersection: IntersectionSummary) -> None:
    labels = [m.method for m in intersection.methods]
    passed = [m.passed_on_intersection for m in intersection.methods]
    failed = [m.failed_on_intersection for m in intersection.methods]
    ax.bar(labels, passed, color="#2ca02c", label="PASS")
    ax.bar(labels, failed, bottom=passed, color="#d62728", label="FAIL")
    ax.set_title("Intersection Pass/Fail Counts By Method")
    ax.set_ylabel("Question count")
    ax.tick_params(axis="x", rotation=20)
    ax.legend()


def _write_summary_md(
    output_dir: Path,
    summaries: list[PassFailSummary],
    chunks: list[ChunkHealth],
    intersection: IntersectionSummary | None,
) -> Path:
    # Persist a text-first artifact so metrics are still accessible in non-GUI environments.
    lines = [
        "# Pass/Fail Dashboard Summary",
        "",
        "## Method Metrics",
        "",
        "| method | pass | fail | total | pass_rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for s in summaries:
        lines.append(
            f"| {s.method} | {s.passed} | {s.failed} | {s.total} | {100.0 * s.pass_rate:.2f}% |"
        )

    lines.extend([
        "",
        "## RLMFix Chunk Health",
        "",
        "| kind | chunk | nonempty | total | nonempty_rate |",
        "|---|---|---:|---:|---:|",
    ])
    if not chunks:
        lines.append("| n/a | n/a | 0 | 0 | 0.00% |")
    else:
        for row in sorted(chunks, key=lambda r: (r.kind, r.chunk)):
            lines.append(
                f"| {row.kind} | {row.chunk} | {row.nonempty} | {row.total} | {100.0 * row.nonempty_rate:.2f}% |"
            )

    if intersection is not None:
        lines.extend(
            [
                "",
                "## Intersection-Only Comparison",
                "",
                f"Model prefix: `{intersection.model_prefix}`",
                "",
                f"Shared question count across baseline/rag/rlm/rlm_rag: **{len(intersection.question_ids)}**",
                "",
                "| method | source_file | total_labeled | pass_on_intersection | fail_on_intersection | pass@1_on_intersection |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for metric in intersection.methods:
            lines.append(
                f"| {metric.method} | {metric.source_file.name} | {metric.total_labeled} | "
                f"{metric.passed_on_intersection} | {metric.failed_on_intersection} | "
                f"{100.0 * metric.pass_rate_on_intersection:.2f}% |"
            )

    out_path = output_dir / "passfail_dashboard_summary.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot pass/fail graphs from stored run artifacts")
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "runs",
        help="Directory containing *_passfail.tsv files.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "raw" / "livecodebench",
        help="Directory containing rlmfix progress files under run folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "visualizations",
        help="Directory for chart and markdown outputs.",
    )
    parser.add_argument(
        "--pattern",
        default="*passfail.tsv",
        help="Glob pattern used inside runs-dir to find pass/fail files.",
    )
    parser.add_argument(
        "--intersection-only",
        action="store_true",
        help=(
            "Add intersection-only comparison plots/tables for baseline/rag/rlm/rlm_rag "
            "using only questions all four methods labeled."
        ),
    )
    parser.add_argument(
        "--intersection-model-prefix",
        default="Qwen2.5-Coder-Ins-32B-",
        help=(
            "Filename prefix used to pick method pass/fail files for intersection mode. "
            "Example: Qwen2.5-Coder-Ins-32B-"
        ),
    )
    parser.add_argument(
        "--intersection-partial-policy",
        choices=["prefer_non_partial", "prefer_partial"],
        default="prefer_non_partial",
        help=(
            "How intersection mode selects method source files when both partial and non-partial "
            "TSVs exist for a method."
        ),
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

    passfail_files = sorted(args.runs_dir.glob(args.pattern))
    if not passfail_files:
        raise SystemExit(f"No pass/fail files found in {args.runs_dir} with pattern {args.pattern}")

    summaries = [_read_passfail(path) for path in passfail_files]
    chunk_rows = _read_chunk_health(args.raw_dir)
    intersection_summary: IntersectionSummary | None = None
    if args.intersection_only:
        intersection_summary = _compute_intersection_summary(
            summaries,
            args.intersection_model_prefix,
            args.intersection_partial_policy,
        )
        print(
            f"Intersection questions (baseline/rag/rlm/rlm_rag): {len(intersection_summary.question_ids)}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    chart_path = args.output_dir / "passfail_dashboard.png"
    try:
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]

        if intersection_summary is None:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            # Fixed 2x2 layout keeps dashboard output stable for downstream sharing.
            _plot_pass_rate(axes[0, 0], summaries)
            _plot_stacked_counts(axes[0, 1], summaries)
            image = _plot_overlap(axes[1, 0], summaries)
            _plot_chunk_health(axes[1, 1], chunk_rows)
            fig.colorbar(image, ax=axes[1, 0], fraction=0.046, pad=0.04)
        else:
            fig, axes = plt.subplots(3, 2, figsize=(16, 17))
            _plot_pass_rate(axes[0, 0], summaries)
            _plot_stacked_counts(axes[0, 1], summaries)
            image = _plot_overlap(axes[1, 0], summaries)
            _plot_chunk_health(axes[1, 1], chunk_rows)
            _plot_intersection_pass_rate(axes[2, 0], intersection_summary)
            _plot_intersection_counts(axes[2, 1], intersection_summary)
            fig.colorbar(image, ax=axes[1, 0], fraction=0.046, pad=0.04)
        fig.suptitle("LiveCodeBench Pass/Fail Dashboard", fontsize=14)
        fig.tight_layout()
        fig.savefig(chart_path, dpi=180)
        plt.close(fig)
        print(f"Wrote chart: {chart_path}")
    except Exception:
        # Markdown summary is still emitted so the script remains useful without plotting deps.
        print(
            "matplotlib not available; skipped PNG chart. "
            "Install with `uv sync --extra results` or `uv pip install matplotlib`."
        )

    summary_path = _write_summary_md(args.output_dir, summaries, chunk_rows, intersection_summary)
    print(f"Wrote summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
