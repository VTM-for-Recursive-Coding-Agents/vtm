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

CREATE TABLE embedding_index_entries (
    memory_id TEXT NOT NULL,
    adapter_id TEXT NOT NULL,
    content_digest TEXT NOT NULL,
    vector_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    data TEXT NOT NULL,
    PRIMARY KEY (memory_id, adapter_id)
);
CREATE INDEX idx_embedding_index_entries_adapter_id
    ON embedding_index_entries(adapter_id);
CREATE INDEX idx_embedding_index_entries_content_digest
    ON embedding_index_entries(content_digest);

INSERT INTO embedding_index_entries (
    memory_id,
    adapter_id,
    content_digest,
    vector_json,
    created_at,
    updated_at,
    data
) VALUES (
    'mem_fixture',
    'deterministic_hash:64',
    'fixture-digest',
    '[0.1, 0.2, 0.3]',
    '2026-04-03T12:00:00Z',
    '2026-04-03T12:00:00Z',
    '{"schema_version":"1.0","memory_id":"mem_fixture","adapter_id":"deterministic_hash:64","content_digest":"fixture-digest","vector":[0.1,0.2,0.3],"created_at":"2026-04-03T12:00:00Z","updated_at":"2026-04-03T12:00:00Z"}'
);

COMMIT;
