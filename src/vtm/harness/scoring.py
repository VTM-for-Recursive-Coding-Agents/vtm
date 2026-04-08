"""Scoring helpers for coding benchmark outputs."""

from __future__ import annotations

from difflib import SequenceMatcher


def changed_path_metrics(
    *,
    expected_changed_paths: tuple[str, ...],
    produced_changed_paths: tuple[str, ...],
) -> tuple[float, float, float]:
    """Return precision, recall, and F1 for changed-path predictions."""
    expected = set(expected_changed_paths)
    produced = set(produced_changed_paths)
    if not expected and not produced:
        return 1.0, 1.0, 1.0
    if not produced:
        return 0.0, 0.0, 0.0
    true_positives = len(expected & produced)
    precision = true_positives / len(produced)
    recall = 0.0 if not expected else true_positives / len(expected)
    if precision + recall == 0.0:
        return precision, recall, 0.0
    return precision, recall, 2.0 * precision * recall / (precision + recall)


def patch_similarity(target_patch: str, produced_patch: str) -> float:
    """Return a rough patch similarity score using sequence matching."""
    return SequenceMatcher(a=target_patch, b=produced_patch).ratio()
