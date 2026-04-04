BEGIN TRANSACTION;

CREATE TABLE schema_meta (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    schema_version INTEGER NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);
INSERT INTO schema_meta (singleton, schema_version, updated_at)
VALUES (1, 1, '2026-04-03T12:00:00Z');
INSERT INTO schema_migrations (version, applied_at)
VALUES (1, '2026-04-03T12:00:00Z');

CREATE TABLE cache_entries (
    digest TEXT PRIMARY KEY,
    tool_name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT,
    data TEXT NOT NULL
);
CREATE INDEX idx_cache_entries_tool_name ON cache_entries(tool_name);

INSERT INTO cache_entries (
    digest,
    tool_name,
    created_at,
    expires_at,
    data
) VALUES (
    'ec4d3df604a500aa7bb14a646f219c7916ea8deef8bed1bb20aa34256dd1b2ea',
    'parser',
    '2026-04-03T12:00:00Z',
    NULL,
    '{"schema_version":"1.0","entry_id":"cache_fixture","key":{"schema_version":"1.0","tool_name":"parser","normalized_args_json":"{\"mode\":\"fixture\"}","repo_fingerprint":{"schema_version":"1.0","repo_root":"/tmp/repo","branch":"main","head_commit":"abc123","tree_digest":"tree-1","dirty_digest":"dirty-1"},"env_fingerprint":{"schema_version":"1.0","python_version":"3.12.8","platform":"darwin-arm64","tool_versions":[{"schema_version":"1.0","name":"pytest","version":"8.3.4"}]},"digest":"ec4d3df604a500aa7bb14a646f219c7916ea8deef8bed1bb20aa34256dd1b2ea"},"value":{"answer":42},"freshness_mode":"prefer_fresh","created_at":"2026-04-03T12:00:00Z","expires_at":null,"hit_count":0}'
);

COMMIT;
