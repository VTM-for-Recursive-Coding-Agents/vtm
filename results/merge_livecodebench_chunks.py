#!/usr/bin/env python3
"""Merge chunk output JSONs from LiveCodeBench into a single provider artifact.

Usage examples
--------------
Merge two chunks into a single file:
  python results/merge_livecodebench_chunks.py \\
      benchmarks/LiveCodeBench/output/Qwen2.5-Coder-Ins-32B-baseline/Scenario.codegeneration_1_0.2_chunk_0_499.json \\
      benchmarks/LiveCodeBench/output/Qwen2.5-Coder-Ins-32B-baseline/Scenario.codegeneration_1_0.2_chunk_500_1055.json \\
      --output benchmarks/LiveCodeBench/output/Qwen2.5-Coder-Ins-32B-baseline/Scenario.codegeneration_1_0.2_merged.json

Glob-style merge (all chunks for a provider):
  python results/merge_livecodebench_chunks.py \\
      benchmarks/LiveCodeBench/output/Qwen2.5-Coder-Ins-32B-baseline/Scenario.codegeneration_1_0.2_chunk_*.json \\
      --output benchmarks/LiveCodeBench/output/Qwen2.5-Coder-Ins-32B-baseline/Scenario.codegeneration_1_0.2_merged.json

After merging, run evaluation with LiveCodeBench:
  cd benchmarks/LiveCodeBench && python -m lcb_runner.runner.custom_evaluator \\
      --custom_output_file <merged_output.json>
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path


def _load_chunk(path: Path) -> list[dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"Failed to read chunk file {path}: {exc}") from exc

    if not isinstance(data, list):
        raise SystemExit(
            f"Chunk file {path} does not contain a JSON array (got {type(data).__name__})."
        )
    return data


def merge_chunks(
    input_paths: list[Path],
    *,
    strict: bool = True,
) -> tuple[list[dict], dict[str, object]]:
    """Merge chunk output lists.

    Returns
    -------
    merged : list[dict]
        Combined entries sorted by question_id.
    provenance : dict
        Summary statistics for logging/auditing.
    """
    seen: dict[str, tuple[int, dict]] = {}  # question_id -> (chunk_idx, entry)
    duplicates: list[dict] = []
    total_loaded = 0

    for chunk_idx, path in enumerate(input_paths):
        entries = _load_chunk(path)
        total_loaded += len(entries)
        for entry in entries:
            if not isinstance(entry, dict):
                raise SystemExit(
                    f"Chunk file {path}: expected each array element to be an object, "
                    f"got {type(entry).__name__}."
                )
            qid = entry.get("question_id")
            if qid is None:
                raise SystemExit(
                    f"Chunk file {path}: entry missing 'question_id' field. "
                    "Cannot merge without a unique key."
                )
            if qid in seen:
                dup_info = {
                    "question_id": qid,
                    "first_chunk_idx": seen[qid][0],
                    "first_chunk_path": str(input_paths[seen[qid][0]]),
                    "duplicate_chunk_idx": chunk_idx,
                    "duplicate_chunk_path": str(path),
                }
                duplicates.append(dup_info)
            else:
                seen[qid] = (chunk_idx, entry)

    if duplicates and strict:
        dup_ids = [d["question_id"] for d in duplicates[:5]]
        extra = f" (and {len(duplicates) - 5} more)" if len(duplicates) > 5 else ""
        raise SystemExit(
            f"Duplicate question_ids detected across chunks: {dup_ids}{extra}. "
            "Chunks must be disjoint. Pass --no-strict to keep the first occurrence."
        )

    merged = [entry for _, entry in seen.values()]
    merged.sort(key=lambda e: str(e.get("question_id", "")))

    provenance = {
        "chunk_count": len(input_paths),
        "chunk_paths": [str(p) for p in input_paths],
        "total_entries_loaded": total_loaded,
        "unique_entries_merged": len(merged),
        "duplicate_entries_dropped": len(duplicates),
        "duplicates": duplicates[:20],  # cap diagnostic output
    }
    return merged, provenance


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge LiveCodeBench chunk output JSONs into a single artifact.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "chunks",
        nargs="+",
        metavar="CHUNK_JSON",
        help="Chunk output JSON files to merge. Supports shell globs.",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        metavar="OUTPUT_JSON",
        help="Path to write the merged combined JSON.",
    )
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="Allow duplicate question_ids (keep first occurrence). Default: fail on duplicates.",
    )
    parser.add_argument(
        "--provenance",
        metavar="PROVENANCE_JSON",
        default=None,
        help="Optional path to write merge provenance/diagnostics as JSON.",
    )
    args = parser.parse_args()

    # Expand any shell globs that the shell did not expand (e.g. when quoted)
    raw_paths: list[Path] = []
    for pattern in args.chunks:
        expanded = glob.glob(pattern)
        if expanded:
            raw_paths.extend(Path(p) for p in sorted(expanded))
        else:
            raw_paths.append(Path(pattern))

    # Validate inputs exist
    missing = [p for p in raw_paths if not p.exists()]
    if missing:
        for p in missing:
            print(f"ERROR: chunk file not found: {p}", file=sys.stderr)
        raise SystemExit(1)

    if not raw_paths:
        raise SystemExit("No input chunk files provided.")

    print(f"[merge] Merging {len(raw_paths)} chunk file(s)...")
    for p in raw_paths:
        print(f"  {p}")

    merged, provenance = merge_chunks(raw_paths, strict=not args.no_strict)

    if provenance["duplicate_entries_dropped"]:
        print(
            f"[merge] WARNING: {provenance['duplicate_entries_dropped']} duplicate entries dropped.",
            file=sys.stderr,
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(merged, indent=4), encoding="utf-8")

    if args.provenance:
        prov_path = Path(args.provenance)
        prov_path.parent.mkdir(parents=True, exist_ok=True)
        prov_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        print(f"[merge] Provenance written to: {prov_path}")

    print(
        f"[merge] Merged {provenance['unique_entries_merged']} entries "
        f"from {provenance['chunk_count']} chunks -> {output_path}"
    )


if __name__ == "__main__":
    main()
