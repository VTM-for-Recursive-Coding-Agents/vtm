# ADR 0011: Docker-Sandboxed Shell Benchmarks

## Status

Accepted

## Context

Repeated attempts and `pass@k` made the coding benchmark runner more credible,
but the harness still lacked two important properties for terminal-benchmark-like
evaluation:

- a built-in sandboxed workspace backend
- tasks whose intended solution path is command execution rather than direct patching

The repo already had a stronger patch-oriented `terminal-smoke` track. The next
step was to add shell-command tasks without creating a second top-level suite or
discarding the existing task-pack and artifact contracts.

## Decision

- Keep shell-command tasks under `suite="coding"`.
- Add `docker_workspace` as a public built-in harness backend alongside
  `local_workspace`.
- Run one long-lived container per attempt with:
  - bind-mounted workspace and artifact roots
  - `--network none` by default
  - `--cap-drop ALL`
  - `--security-opt no-new-privileges`
  - writable `tmpfs` at `/tmp`
- Extend `HarnessTaskPack` with `execution_style` and use
  `execution_style="shell_command"` for the new shell track.
- Record normalized backend and Docker metadata in `ExecutorRequest`,
  `ExecutorResult`, `attempts.jsonl`, `results.jsonl`, and `manifest.lock.json`.
- Add `benchmarks/manifests/terminal-shell-smoke.json` as the checked-in shell
  track, backed by the same synthetic terminal corpus family.
- Keep `results.jsonl` as one aggregate row per case and `attempts.jsonl` as one
  row per attempt.
- For native-agent shell-command tasks, enforce
  `tool_policy="no_file_mutation"` so terminal, read/search, and memory tools
  remain available while direct file-mutation tools are withheld.

## Consequences

- The harness now has a built-in sandboxed backend that stays inspectable and
  local-first.
- Shell-command tasks can be evaluated with the same attempt-aware reporting and
  diff-based scoring used by existing coding tasks.
- Reporting now breaks coding runs down by both `execution_style` and
  `workspace_backend`.
- The design remains intentionally narrow:
  - Docker is the only built-in sandbox backend in this pass.
  - Shell-command tasks still use the existing `test_command` verifier.
  - There is still no separate shell-only suite, remote sandbox executor, or
    multi-agent runtime.
