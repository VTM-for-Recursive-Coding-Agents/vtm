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

CREATE TABLE artifacts (
    artifact_id TEXT PRIMARY KEY,
    sha256 TEXT NOT NULL,
    relative_path TEXT NOT NULL,
    data TEXT NOT NULL
);
CREATE INDEX idx_artifacts_sha ON artifacts(sha256);

INSERT INTO artifacts (
    artifact_id,
    sha256,
    relative_path,
    data
) VALUES (
    'art_fixture_v1',
    '34f7aed3bc21db8ad882cdc561813afe29bea539f2f951568c38d2c98c2c75ca',
    'sha256/34f7aed3bc21db8ad882cdc561813afe29bea539f2f951568c38d2c98c2c75ca',
    '{"schema_version":"1.0","artifact_id":"art_fixture_v1","sha256":"34f7aed3bc21db8ad882cdc561813afe29bea539f2f951568c38d2c98c2c75ca","relative_path":"sha256/34f7aed3bc21db8ad882cdc561813afe29bea539f2f951568c38d2c98c2c75ca","size_bytes":16,"content_type":"text/plain","tool_name":"fixture-tool","tool_version":null,"capture_state":"committed","capture_group_id":null,"actor":"system","session_id":null,"created_at":"2026-04-03T12:00:00Z","committed_at":null,"abandoned_at":null,"metadata":{"capture":"fixture"}}'
);

COMMIT;
