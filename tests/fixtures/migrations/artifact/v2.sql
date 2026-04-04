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
VALUES (1, 2, '2026-04-03T12:00:00Z');
INSERT INTO schema_migrations (version, applied_at)
VALUES (1, '2026-04-03T12:00:00Z');
INSERT INTO schema_migrations (version, applied_at)
VALUES (2, '2026-04-03T12:01:00Z');

CREATE TABLE artifacts (
    artifact_id TEXT PRIMARY KEY,
    sha256 TEXT NOT NULL,
    relative_path TEXT NOT NULL,
    capture_state TEXT NOT NULL DEFAULT 'committed',
    capture_group_id TEXT,
    committed_at TEXT,
    abandoned_at TEXT,
    actor TEXT NOT NULL DEFAULT 'system',
    session_id TEXT,
    data TEXT NOT NULL
);
CREATE INDEX idx_artifacts_sha ON artifacts(sha256);
CREATE INDEX idx_artifacts_state ON artifacts(capture_state);
CREATE INDEX idx_artifacts_group ON artifacts(capture_group_id);

INSERT INTO artifacts (
    artifact_id,
    sha256,
    relative_path,
    capture_state,
    capture_group_id,
    committed_at,
    abandoned_at,
    actor,
    session_id,
    data
) VALUES (
    'art_fixture_v2',
    '97a5a40a244a405e7f7afc0deba90bf19c65f0defb7595f4ddfb50e122e40bae',
    'sha256/97a5a40a244a405e7f7afc0deba90bf19c65f0defb7595f4ddfb50e122e40bae',
    'committed',
    'grp_fixture',
    '2026-04-03T12:00:00Z',
    NULL,
    'fixture-tests',
    'sess_fixture',
    '{"schema_version":"1.0","artifact_id":"art_fixture_v2","sha256":"97a5a40a244a405e7f7afc0deba90bf19c65f0defb7595f4ddfb50e122e40bae","relative_path":"sha256/97a5a40a244a405e7f7afc0deba90bf19c65f0defb7595f4ddfb50e122e40bae","size_bytes":20,"content_type":"text/plain","tool_name":"fixture-tool","tool_version":null,"capture_state":"committed","capture_group_id":"grp_fixture","actor":"fixture-tests","session_id":"sess_fixture","created_at":"2026-04-03T12:00:00Z","committed_at":"2026-04-03T12:00:00Z","abandoned_at":null,"metadata":{"capture":"fixture-v2"}}'
);

COMMIT;
