"""Embedding adapter contracts and a deterministic local implementation."""

from __future__ import annotations

import hashlib
import math
from typing import Protocol


class EmbeddingAdapter(Protocol):
    """Interface for text-to-vector adapters used by retrieval."""

    @property
    def adapter_id(self) -> str: ...

    def embed_text(self, text: str) -> tuple[float, ...]: ...


class DeterministicHashEmbeddingAdapter:
    """Cheap deterministic embedding adapter for local testing and benchmarks."""

    def __init__(self, *, dimensions: int = 64) -> None:
        """Create a deterministic embedding adapter with a fixed vector width."""
        if dimensions <= 0:
            raise ValueError("Deterministic hash embedding adapter requires dimensions > 0")
        self._dimensions = dimensions

    @property
    def adapter_id(self) -> str:
        """Stable adapter identifier used by the embedding index."""
        return f"deterministic_hash:{self._dimensions}"

    def embed_text(self, text: str) -> tuple[float, ...]:
        """Project text into a normalized hashed feature vector."""
        normalized = text.strip().lower()
        if not normalized:
            return tuple(0.0 for _ in range(self._dimensions))

        features = self._features(normalized)
        if not features:
            return tuple(0.0 for _ in range(self._dimensions))

        vector = [0.0] * self._dimensions
        for feature in features:
            digest = hashlib.sha256(feature.encode("utf-8")).digest()
            index = int.from_bytes(digest[:8], "big") % self._dimensions
            sign = 1.0 if digest[8] % 2 == 0 else -1.0
            weight = 1.0 + (digest[9] / 255.0)
            vector[index] += sign * weight

        norm = math.sqrt(sum(component * component for component in vector))
        if norm == 0.0:
            return tuple(0.0 for _ in range(self._dimensions))
        return tuple(component / norm for component in vector)

    def _features(self, text: str) -> tuple[str, ...]:
        words = text.split()
        if not words:
            return ()
        features: list[str] = []
        for word in words:
            features.append(f"tok:{word}")
            if len(word) < 3:
                features.append(f"chr:{word}")
                continue
            padded = f"  {word}  "
            for index in range(len(padded) - 2):
                features.append(f"tri:{padded[index:index + 3]}")
        return tuple(features)
