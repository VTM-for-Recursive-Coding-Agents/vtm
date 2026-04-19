"""Load public LiveCodeBench problems through the checkout's local venv."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Final

from huggingface_hub import hf_hub_download

DEFAULT_DATASET_REPO: Final[str] = "livecodebench/code_generation_lite"
DEFAULT_DATASET_FILENAME: Final[str] = "test.jsonl"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch a bounded set of public LiveCodeBench problems as JSON.",
    )
    parser.add_argument("--dataset-repo", default=DEFAULT_DATASET_REPO)
    parser.add_argument("--dataset-filename", default=DEFAULT_DATASET_FILENAME)
    parser.add_argument("--problem-offset", type=int, default=0)
    parser.add_argument("--max-problems", type=int, required=True)
    return parser


def load_rows(
    *,
    dataset_repo: str,
    dataset_filename: str,
    problem_offset: int,
    max_problems: int,
) -> list[dict[str, object]]:
    dataset_path = Path(
        hf_hub_download(
            repo_id=dataset_repo,
            repo_type="dataset",
            filename=dataset_filename,
        )
    )
    rows: list[dict[str, object]] = []
    with dataset_path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            stripped = line.strip()
            if not stripped:
                continue
            if index < problem_offset:
                continue
            payload = json.loads(stripped)
            if isinstance(payload, dict):
                rows.append(payload)
            if len(rows) >= max_problems:
                break
    return rows


def main() -> int:
    args = build_parser().parse_args()
    rows = load_rows(
        dataset_repo=args.dataset_repo,
        dataset_filename=args.dataset_filename,
        problem_offset=max(0, int(args.problem_offset)),
        max_problems=max(1, int(args.max_problems)),
    )
    print(json.dumps(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
