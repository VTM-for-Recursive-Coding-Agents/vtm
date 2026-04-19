#!/usr/bin/env python3
"""Run an optional LongCoT pilot against OpenRouter."""

from __future__ import annotations

import sys
from pathlib import Path


def _main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from vtm.benchmarks.longcot_pilot import main

    return main()


if __name__ == "__main__":
    raise SystemExit(_main())
