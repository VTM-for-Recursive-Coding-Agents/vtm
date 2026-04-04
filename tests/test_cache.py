from __future__ import annotations

from datetime import UTC, datetime, timedelta

from vtm.cache import CacheEntry, CacheKey
from vtm.enums import FreshnessMode


def test_cache_key_is_deterministic(repo_fp, env_fp) -> None:
    key_one = CacheKey.from_parts("tool", {"b": 2, "a": 1}, repo_fp, env_fp)
    key_two = CacheKey.from_parts("tool", {"a": 1, "b": 2}, repo_fp, env_fp)
    changed_repo = repo_fp.model_copy(update={"dirty_digest": "dirty-2"})
    changed_key = CacheKey.from_parts("tool", {"a": 1, "b": 2}, changed_repo, env_fp)

    assert key_one.digest == key_two.digest
    assert key_one.normalized_args_json == key_two.normalized_args_json
    assert changed_key.digest != key_one.digest


def test_cache_hit_miss_logging(cache_store, metadata_store, repo_fp, env_fp) -> None:
    key = CacheKey.from_parts("tool", {"path": "src/main.py"}, repo_fp, env_fp)
    miss_key = CacheKey.from_parts("tool", {"path": "src/other.py"}, repo_fp, env_fp)
    cache_store.save_cache_entry(CacheEntry(key=key, value={"answer": 42}))

    hit = cache_store.get_cache_entry(key)
    miss = cache_store.get_cache_entry(miss_key)
    event_types = [event.event_type for event in metadata_store.list_events()]

    assert hit is not None
    assert hit.hit_count == 1
    assert miss is None
    assert event_types == ["cache_hit", "cache_miss"]


def test_cache_get_non_expired_entry_with_expiry_does_not_raise(
    cache_store,
    repo_fp,
    env_fp,
) -> None:
    key = CacheKey.from_parts("tool", {"path": "src/live.py"}, repo_fp, env_fp)
    cache_store.save_cache_entry(
        CacheEntry(
            key=key,
            value={"answer": 42},
            expires_at=datetime.now(UTC) + timedelta(minutes=5),
        )
    )

    hit = cache_store.get_cache_entry(key)

    assert hit is not None
    assert hit.value == {"answer": 42}


def test_expired_cache_entry_misses_under_default_freshness_mode(
    cache_store,
    repo_fp,
    env_fp,
) -> None:
    key = CacheKey.from_parts("tool", {"path": "src/expired.py"}, repo_fp, env_fp)
    created_at = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    cache_store.save_cache_entry(
        CacheEntry(
            key=key,
            value={"answer": 7},
            created_at=created_at,
            expires_at=created_at + timedelta(minutes=5),
        )
    )

    hit = cache_store.get_cache_entry(key, now=datetime(2026, 1, 1, 12, 6))

    assert hit is None


def test_expired_cache_entry_can_be_returned_when_stale_is_allowed(
    cache_store,
    repo_fp,
    env_fp,
) -> None:
    key = CacheKey.from_parts("tool", {"path": "src/stale.py"}, repo_fp, env_fp)
    created_at = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    cache_store.save_cache_entry(
        CacheEntry(
            key=key,
            value={"answer": 9},
            freshness_mode=FreshnessMode.ALLOW_STALE,
            created_at=created_at,
            expires_at=created_at + timedelta(minutes=5),
        )
    )

    hit = cache_store.get_cache_entry(key, now=datetime(2026, 1, 1, 12, 6))

    assert hit is not None
    assert hit.value == {"answer": 9}
