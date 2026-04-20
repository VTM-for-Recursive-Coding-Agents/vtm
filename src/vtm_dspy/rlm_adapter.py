"""Optional DSPy RLM context helpers for VTM memory cards."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from vtm.fingerprints import DependencyFingerprint
from vtm.memory_items import MemoryItem, VisibilityScope
from vtm.services.memory_kernel import MemoryKernel

from . import require_dspy
from .config import DSPyOpenRouterConfig
from .tools import VTMMemoryTools

RLM_SANDBOX_NOTE = (
    "DSPy RLM defaults to a Deno/Pyodide sandbox for code execution. Use it for "
    "long-context reasoning or tool mediation, not direct repository editing without "
    "a custom interpreter."
)


@dataclass
class VTMRLMContextAdapter:
    """Prepare compact VTM memory cards for DSPy RLM usage."""

    memory_tools: VTMMemoryTools
    model_config: DSPyOpenRouterConfig
    max_cards: int = 5

    @classmethod
    def from_kernel(
        cls,
        *,
        kernel: MemoryKernel | None,
        scopes: Sequence[VisibilityScope] = (),
        dependency_provider: Callable[[], DependencyFingerprint | None] | None = None,
        memory_lookup: Callable[[str], MemoryItem | None] | None = None,
        model_config: DSPyOpenRouterConfig | None = None,
        max_cards: int = 5,
    ) -> VTMRLMContextAdapter:
        """Create an adapter directly from the current kernel and scope configuration."""
        return cls(
            memory_tools=VTMMemoryTools(
                kernel=kernel,
                scopes=scopes,
                dependency_provider=dependency_provider,
                memory_lookup=memory_lookup,
            ),
            model_config=model_config or DSPyOpenRouterConfig.from_env(),
            max_cards=max_cards,
        )

    def memory_cards(self, query: str, *, k: int | None = None) -> list[dict[str, Any]]:
        """Return compact verified memory cards for one DSPy RLM query."""
        return self.memory_tools.search_verified_memory(query, k or self.max_cards)

    def build_context(self, query: str, *, k: int | None = None) -> dict[str, Any]:
        """Build a serializable context bundle for long-context DSPy reasoning."""
        cards = self.memory_cards(query, k=k)
        return {
            "memory_cards": cards,
            "sandbox_note": RLM_SANDBOX_NOTE,
            "tool_names": list(self.memory_tools.tool_mapping().keys()),
            "instructions": (
                "Treat the memory cards as verified repository context when possible. "
                "If a card looks stale or insufficient, call the exposed VTM tools again."
            ),
        }


def make_vtm_rlm(
    *,
    adapter: VTMRLMContextAdapter,
    signature: str = "task, context -> response",
    query: str,
    tools: Sequence[Callable[..., Any]] | None = None,
    **kwargs: Any,
) -> Any:
    """Instantiate `dspy.RLM` with VTM memory tools when DSPy exposes that surface."""
    dspy = require_dspy()
    if not hasattr(dspy, "RLM"):
        raise RuntimeError("installed DSPy does not expose dspy.RLM")
    constructor_kwargs = dict(kwargs)
    constructor_kwargs.setdefault(
        "tools",
        list(tools) if tools is not None else list(adapter.memory_tools.tool_mapping().values()),
    )
    return dspy.RLM(signature, **constructor_kwargs)


__all__ = ["RLM_SANDBOX_NOTE", "VTMRLMContextAdapter", "make_vtm_rlm"]
