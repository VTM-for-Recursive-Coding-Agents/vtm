"""Helpers for importing the vendored upstream RLM runtime."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any


def vendored_rlm_root() -> Path:
    """Return the repository-local root for the vendored upstream RLM."""
    return Path(__file__).resolve().parents[2] / "vendor" / "rlm"


def ensure_vendored_rlm_on_path() -> Path:
    """Prepend the vendored upstream RLM root to `sys.path`."""
    root = vendored_rlm_root()
    if not root.exists():
        raise RuntimeError(
            "vendored RLM runtime is missing; expected to find it under vendor/rlm"
        )
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def load_rlm_runtime() -> tuple[type[Any], type[Any]]:
    """Load the vendored `RLM` and `RLMLogger` symbols with a clear dependency error."""
    ensure_vendored_rlm_on_path()
    try:
        rlm_module = importlib.import_module("rlm")
        logger_module = importlib.import_module("rlm.logger")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "vendored RLM dependencies are missing; install the repository with the "
            "`rlm` extra to provide openai, python-dotenv, and rich"
        ) from exc
    return rlm_module.RLM, logger_module.RLMLogger
