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

CREATE TABLE memory_items (
    memory_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    scope_kind TEXT NOT NULL,
    scope_id TEXT NOT NULL,
    tx_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    data TEXT NOT NULL
);
CREATE INDEX idx_memory_items_status ON memory_items(status);
CREATE INDEX idx_memory_items_scope ON memory_items(scope_kind, scope_id);
CREATE INDEX idx_memory_items_tx ON memory_items(tx_id);

CREATE TABLE lineage_edges (
    parent_id TEXT NOT NULL,
    child_id TEXT NOT NULL,
    edge_type TEXT NOT NULL,
    tx_id TEXT,
    created_at TEXT NOT NULL,
    data TEXT NOT NULL,
    PRIMARY KEY (parent_id, child_id, edge_type, created_at)
);
CREATE INDEX idx_lineage_edges_child ON lineage_edges(child_id);
CREATE INDEX idx_lineage_edges_tx ON lineage_edges(tx_id);

CREATE TABLE transactions (
    tx_id TEXT PRIMARY KEY,
    parent_tx_id TEXT,
    state TEXT NOT NULL,
    scope_kind TEXT NOT NULL,
    scope_id TEXT NOT NULL,
    opened_at TEXT NOT NULL,
    committed_at TEXT,
    rolled_back_at TEXT,
    data TEXT NOT NULL
);
CREATE INDEX idx_transactions_parent ON transactions(parent_tx_id);
CREATE INDEX idx_transactions_state ON transactions(state);

CREATE TABLE staged_memory_items (
    tx_id TEXT NOT NULL,
    stage_order INTEGER NOT NULL,
    memory_id TEXT NOT NULL,
    data TEXT NOT NULL,
    PRIMARY KEY (tx_id, stage_order)
);
CREATE INDEX idx_staged_memory_items_tx_order
    ON staged_memory_items(tx_id, stage_order);

CREATE TABLE events (
    event_id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,
    occurred_at TEXT NOT NULL,
    tx_id TEXT,
    memory_id TEXT,
    cache_digest TEXT,
    exported_to_jsonl INTEGER NOT NULL DEFAULT 0,
    data TEXT NOT NULL
);
CREATE INDEX idx_events_type ON events(event_type);
CREATE INDEX idx_events_tx ON events(tx_id);
CREATE INDEX idx_events_memory ON events(memory_id);
CREATE INDEX idx_events_export ON events(exported_to_jsonl, occurred_at, event_id);

INSERT INTO events (
    event_id,
    event_type,
    occurred_at,
    tx_id,
    memory_id,
    cache_digest,
    exported_to_jsonl,
    data
) VALUES (
    'evt_fixture',
    'fixture_event',
    '2026-04-03T12:00:00Z',
    NULL,
    NULL,
    NULL,
    0,
    '{"schema_version":"1.0","event_id":"evt_fixture","event_type":"fixture_event","occurred_at":"2026-04-03T12:00:00Z","tx_id":null,"memory_id":null,"cache_digest":null,"actor":"system","session_id":null,"tool_name":null,"payload":{}}'
);

COMMIT;
