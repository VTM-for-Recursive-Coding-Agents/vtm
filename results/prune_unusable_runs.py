#!/usr/bin/env python3
"""Audit and optionally delete unusable benchmark run folders."""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path

from normalize_results import get_unusable_reasons, parse_run_dir


@dataclass
class PruneCandidate:
    root: Path
    run_dir: Path
    benchmark: str
    run_id: str
    reasons: list[str]


def _default_roots() -> list[Path]:
    base = Path(__file__).resolve().parent
    return [base / "raw", base / "archive"]


def _discover_run_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted({metadata_path.parent for metadata_path in root.rglob("metadata.txt")})


def collect_prune_candidates(
    roots: list[Path],
    *,
    known_failed_run_ids: set[str] | None = None,
) -> list[PruneCandidate]:
    candidates: list[PruneCandidate] = []

    for root in roots:
        for run_dir in _discover_run_dirs(root):
            record = parse_run_dir(run_dir)
            if record is None:
                continue
            reasons = get_unusable_reasons(record, known_failed_run_ids=known_failed_run_ids)
            if not reasons:
                continue
            candidates.append(
                PruneCandidate(
                    root=root,
                    run_dir=run_dir,
                    benchmark=record.benchmark,
                    run_id=record.run_id,
                    reasons=reasons,
                )
            )

    return sorted(candidates, key=lambda item: (str(item.root), item.benchmark, item.run_id, str(item.run_dir)))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit and optionally delete unusable benchmark run folders")
    parser.add_argument(
        "--root",
        type=Path,
        action="append",
        default=[],
        help="Root to scan for run folders. Defaults to results/raw and results/archive.",
    )
    parser.add_argument(
        "--known-failed-run-id",
        action="append",
        default=[],
        help="Run id to treat as unusable even if parsing would otherwise consider it usable. Repeat as needed.",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete the audited unusable run folders. Without this flag, the command is dry-run only.",
    )
    return parser


def _print_candidates(candidates: list[PruneCandidate]) -> None:
    if not candidates:
        print("[prune] No unusable run folders found.")
        return

    print(f"[prune] Candidates: {len(candidates)}")
    for candidate in candidates:
        location = candidate.run_dir.relative_to(candidate.root)
        reason_text = ", ".join(candidate.reasons)
        print(
            f"  - root={candidate.root} benchmark={candidate.benchmark} run_id={candidate.run_id} "
            f"path={location} reasons={reason_text}"
        )


def _delete_candidates(candidates: list[PruneCandidate]) -> None:
    for candidate in candidates:
        shutil.rmtree(candidate.run_dir)
        print(f"[prune] Deleted {candidate.run_dir}")


def main() -> int:
    args = build_arg_parser().parse_args()
    roots = args.root or _default_roots()
    known_failed_run_ids = set(args.known_failed_run_id)
    candidates = collect_prune_candidates(roots, known_failed_run_ids=known_failed_run_ids)

    _print_candidates(candidates)
    if not candidates:
        return 0

    if not args.delete:
        print("[prune] Dry run only. Re-run with --delete to remove these folders.")
        return 0

    _delete_candidates(candidates)
    print(f"[prune] Deleted {len(candidates)} run folders.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())