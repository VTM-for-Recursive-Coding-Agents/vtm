#!/usr/bin/env python3
"""CLI entrypoint for the small LiveCodeBench DSPy plus VTM pilot."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

if __name__ == "__main__":
    from vtm.benchmarks.livecodebench_dspy_pilot import main

    raise SystemExit(main())
