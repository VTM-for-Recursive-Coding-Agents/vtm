"""Optional DSPy integration surface for VTM."""

from __future__ import annotations

import importlib
from importlib import import_module
from typing import Any

OPTIONAL_DEPENDENCY_MESSAGE = (
    "vtm_dspy requires the optional 'dspy' dependency for DSPy runtime features. "
    "Install it with `uv sync --extra dspy` or `pip install dspy`."
)

_EXPORT_TO_MODULE = {
    "DEFAULT_DSPY_MODEL": "vtm_dspy.config",
    "DSPyOpenRouterConfig": "vtm_dspy.config",
    "resolve_dspy_model": "vtm_dspy.config",
    "resolve_dspy_lm_model": "vtm_dspy.config",
    "VTMReActCodingAgent": "vtm_dspy.react_agent",
    "VTMMemoryTools": "vtm_dspy.tools",
    "WorkspaceTools": "vtm_dspy.tools",
}

__all__ = [
    "DEFAULT_DSPY_MODEL",
    "DSPyOpenRouterConfig",
    "OPTIONAL_DEPENDENCY_MESSAGE",
    "VTMReActCodingAgent",
    "VTMMemoryTools",
    "WorkspaceTools",
    "dspy_available",
    "require_dspy",
    "resolve_dspy_lm_model",
    "resolve_dspy_model",
]


def dspy_available() -> bool:
    """Return whether the optional DSPy runtime is importable."""
    try:
        importlib.import_module("dspy")
    except ImportError:
        return False
    return True


def require_dspy() -> Any:
    """Import DSPy with a clear optional-dependency error."""
    try:
        return importlib.import_module("dspy")
    except ImportError as exc:  # pragma: no cover - exercised in tests via patched import
        raise ImportError(OPTIONAL_DEPENDENCY_MESSAGE) from exc


def __getattr__(name: str) -> Any:
    """Lazily load optional integration helpers without importing DSPy eagerly."""
    module_name = _EXPORT_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module 'vtm_dspy' has no attribute {name!r}")
    module = import_module(module_name)
    return getattr(module, name)
