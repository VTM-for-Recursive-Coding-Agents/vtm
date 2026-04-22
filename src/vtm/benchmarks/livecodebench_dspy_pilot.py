"""Compatibility loader for the experimental LiveCodeBench DSPy pilot."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_EXPERIMENT_PATH = (
    Path(__file__).resolve().parents[3] / "experiments" / "livecodebench_dspy_pilot.py"
)
_SPEC = importlib.util.spec_from_file_location(__name__, _EXPERIMENT_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Unable to load experimental module from {_EXPERIMENT_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[__name__] = _MODULE
_SPEC.loader.exec_module(_MODULE)
