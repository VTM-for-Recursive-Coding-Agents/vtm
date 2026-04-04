from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from vtm.events import MemoryEvent
from vtm.stores.sqlite_store import SqliteMetadataStore


def test_metadata_events_export_to_jsonl_in_deterministic_order(tmp_path: Path) -> None:
    db_path = tmp_path / "metadata.sqlite"
    event_log_path = tmp_path / "events.jsonl"
    store = SqliteMetadataStore(db_path, event_log_path=event_log_path)
    try:
        early_b = MemoryEvent(
            event_id="evt_b",
            event_type="beta",
            occurred_at=datetime(2026, 4, 3, 12, 0, tzinfo=UTC),
        )
        early_a = MemoryEvent(
            event_id="evt_a",
            event_type="alpha",
            occurred_at=datetime(2026, 4, 3, 12, 0, tzinfo=UTC),
        )
        later = MemoryEvent(
            event_id="evt_c",
            event_type="gamma",
            occurred_at=datetime(2026, 4, 3, 12, 1, tzinfo=UTC),
        )

        for event in (later, early_b, early_a):
            store.save_event(event)

        assert not event_log_path.exists()
        assert store.export_events_to_jsonl() == 3
        assert store.export_events_to_jsonl() == 0

        exported = tuple(
            MemoryEvent.from_json(line)
            for line in event_log_path.read_text(encoding="utf-8").splitlines()
        )
        assert tuple(event.event_id for event in exported) == ("evt_a", "evt_b", "evt_c")

        with sqlite3.connect(db_path) as conn:
            flags = conn.execute(
                """
                SELECT exported_to_jsonl
                FROM events
                ORDER BY occurred_at ASC, event_id ASC
                """
            ).fetchall()
        assert [row[0] for row in flags] == [1, 1, 1]
    finally:
        store.close()


def test_event_export_is_at_least_once_and_rebuild_rewrites_deduped_log(tmp_path: Path) -> None:
    db_path = tmp_path / "metadata.sqlite"
    event_log_path = tmp_path / "events.jsonl"
    store = SqliteMetadataStore(db_path, event_log_path=event_log_path)
    try:
        first = MemoryEvent(
            event_id="evt_a",
            event_type="alpha",
            occurred_at=datetime(2026, 4, 3, 12, 0, tzinfo=UTC),
        )
        second = MemoryEvent(
            event_id="evt_b",
            event_type="beta",
            occurred_at=datetime(2026, 4, 3, 12, 1, tzinfo=UTC),
        )
        store.save_event(first)
        store.save_event(second)

        original_mark = store._mark_event_exported
        seen = 0

        def fail_after_first(event: MemoryEvent) -> None:
            nonlocal seen
            seen += 1
            if seen == 1:
                raise RuntimeError("simulated export cursor failure")
            original_mark(event)

        store._mark_event_exported = fail_after_first  # type: ignore[attr-defined]
        try:
            with pytest.raises(RuntimeError, match="simulated export cursor failure"):
                store.export_events_to_jsonl()
        finally:
            store._mark_event_exported = original_mark  # type: ignore[attr-defined]

        assert event_log_path.read_text(encoding="utf-8").splitlines() == [first.to_json()]
        assert store.export_events_to_jsonl() == 2
        exported_ids = [
            MemoryEvent.from_json(line).event_id
            for line in event_log_path.read_text(encoding="utf-8").splitlines()
        ]
        assert exported_ids == ["evt_a", "evt_a", "evt_b"]

        assert store.rebuild_events_jsonl() == 2
        rebuilt_ids = [
            MemoryEvent.from_json(line).event_id
            for line in event_log_path.read_text(encoding="utf-8").splitlines()
        ]
        assert rebuilt_ids == ["evt_a", "evt_b"]
        state = store.get_event_export_state()
        assert state is not None
        assert state["last_exported_event_id"] == "evt_b"
        assert state["full_rebuild_count"] == 1
    finally:
        store.close()
