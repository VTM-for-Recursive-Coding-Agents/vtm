"""SWE-bench Lite dataset normalization and manifest generation helpers."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from vtm.benchmarks.models import BenchmarkManifest, CodingTaskCase, CommitPair, RepoSpec


@dataclass(frozen=True)
class SWEbenchLiteInstance:
    """Normalized SWE-bench Lite row used by VTM preparation code."""

    instance_id: str
    repo: str
    repo_name: str
    remote_url: str
    base_commit: str
    patch: str
    test_patch: str
    problem_statement: str
    hints_text: str | None
    fail_to_pass_tests: tuple[str, ...]
    pass_to_pass_tests: tuple[str, ...]
    dataset_name: str


class SWEbenchLitePreparer:
    """Loads SWE-bench Lite instances and turns them into VTM manifests."""

    def load_instances(
        self,
        *,
        dataset_name: str,
        dataset_path: str | Path | None = None,
        repo_filters: Sequence[str] = (),
        instance_filters: Sequence[str] = (),
        max_instances: int | None = None,
    ) -> list[SWEbenchLiteInstance]:
        """Load and normalize SWE-bench Lite instances with optional filtering."""
        rows = self._load_rows(dataset_name=dataset_name, dataset_path=dataset_path)
        selected_repo_filters = set(repo_filters)
        selected_instance_filters = set(instance_filters)
        instances: list[SWEbenchLiteInstance] = []
        for row in rows:
            instance = self._normalize_row(row, dataset_name=dataset_name)
            if selected_repo_filters and instance.repo_name not in selected_repo_filters:
                continue
            if selected_instance_filters and instance.instance_id not in selected_instance_filters:
                continue
            instances.append(instance)
            if max_instances is not None and len(instances) >= max_instances:
                break
        missing_repos = sorted(selected_repo_filters - {item.repo_name for item in instances})
        if missing_repos:
            raise ValueError(f"unknown SWE-bench repos: {missing_repos}")
        missing_instances = sorted(
            selected_instance_filters - {item.instance_id for item in instances}
        )
        if missing_instances:
            raise ValueError(f"unknown SWE-bench instances: {missing_instances}")
        return instances

    def prepare_manifest(
        self,
        *,
        dataset_name: str,
        cache_root: str | Path,
        output_manifest: str | Path,
        dataset_path: str | Path | None = None,
        repo_filters: Sequence[str] = (),
        instance_filters: Sequence[str] = (),
        max_instances: int | None = None,
    ) -> BenchmarkManifest:
        """Prepare repos and write a VTM benchmark manifest for SWE-bench Lite."""
        cache_root_path = Path(cache_root)
        instances = self.load_instances(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            repo_filters=repo_filters,
            instance_filters=instance_filters,
            max_instances=max_instances,
        )
        prepared = self.prepare_instances(instances=instances, cache_root=cache_root_path)
        manifest = self.build_manifest(
            dataset_name=dataset_name,
            prepared_instances=prepared,
        )
        output_manifest_path = Path(output_manifest)
        output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        output_manifest_path.write_text(
            manifest.to_json(),
            encoding="utf-8",
        )
        return manifest

    def prepare_instances(
        self,
        *,
        instances: Sequence[SWEbenchLiteInstance],
        cache_root: str | Path,
    ) -> list[PreparedSWEbenchLiteInstance]:
        """Prepare local git refs and metadata for each selected instance."""
        cache_root_path = Path(cache_root)
        repos_root = cache_root_path / "repos"
        worktrees_root = cache_root_path / "worktrees"
        repos_root.mkdir(parents=True, exist_ok=True)
        worktrees_root.mkdir(parents=True, exist_ok=True)
        prepared: list[PreparedSWEbenchLiteInstance] = []
        repo_cache: dict[str, PreparedRepoCache] = {}
        for instance in instances:
            cached_repo = repo_cache.get(instance.repo_name)
            if cached_repo is None:
                cached_repo = self._prepare_repo_cache(instance, repos_root)
                repo_cache[instance.repo_name] = cached_repo
            refs = self._prepare_instance_refs(
                instance=instance,
                cache_root=cache_root_path,
                repo_cache=cached_repo,
            )
            prepared.append(
                PreparedSWEbenchLiteInstance(
                    instance=instance,
                    repo_cache=cached_repo,
                    base_ref=refs.base_ref,
                    gold_ref=refs.gold_ref,
                    expected_changed_paths=self._modified_paths_from_patch(instance.patch),
                    gold_test_patch_digest=self._sha256(instance.test_patch)
                    if instance.test_patch.strip()
                    else None,
                )
            )
        return prepared

    def build_manifest(
        self,
        *,
        dataset_name: str,
        prepared_instances: Sequence[PreparedSWEbenchLiteInstance],
    ) -> BenchmarkManifest:
        """Convert prepared SWE-bench instances into a benchmark manifest."""
        repos: OrderedDict[str, list[PreparedSWEbenchLiteInstance]] = OrderedDict()
        for prepared in prepared_instances:
            repos.setdefault(prepared.instance.repo_name, []).append(prepared)

        repo_specs: list[RepoSpec] = []
        coding_tasks: list[CodingTaskCase] = []
        for repo_name, items in repos.items():
            first = items[0]
            repo_specs.append(
                RepoSpec(
                    repo_name=repo_name,
                    source_kind="git",
                    remote_url=str(first.repo_cache.repo_root),
                    branch=first.repo_cache.default_branch,
                    commit_pairs=tuple(
                        CommitPair(
                            pair_id=item.instance.instance_id,
                            base_ref=item.base_ref,
                            head_ref=item.gold_ref,
                            label=item.instance.instance_id,
                            description=item.instance.problem_statement.strip().splitlines()[0],
                        )
                        for item in items
                    ),
                )
            )
            for item in items:
                coding_tasks.append(
                    CodingTaskCase(
                        case_id=item.instance.instance_id,
                        repo_name=repo_name,
                        commit_pair_id=item.instance.instance_id,
                        evaluation_backend="swebench_harness",
                        instance_id=item.instance.instance_id,
                        dataset_name=dataset_name,
                        task_statement=item.instance.problem_statement.strip(),
                        problem_statement=item.instance.problem_statement.strip(),
                        hints_text=item.instance.hints_text,
                        failing_tests=item.instance.fail_to_pass_tests,
                        fail_to_pass_tests=item.instance.fail_to_pass_tests,
                        pass_to_pass_tests=item.instance.pass_to_pass_tests,
                        touched_paths=item.expected_changed_paths,
                        expected_changed_paths=item.expected_changed_paths,
                        test_command=(),
                        target_patch=item.instance.patch,
                        gold_test_patch_digest=item.gold_test_patch_digest,
                        task_kind="swebench_lite",
                        difficulty="external",
                    )
                )
        return BenchmarkManifest(
            manifest_id="swebench_lite_generated",
            description=(
                "Generated SWE-bench Lite coding benchmark manifest backed by local repo caches."
            ),
            repos=tuple(repo_specs),
            coding_tasks=tuple(coding_tasks),
            seed=0,
        )

    def _load_rows(
        self,
        *,
        dataset_name: str,
        dataset_path: str | Path | None,
    ) -> list[dict[str, object]]:
        if dataset_path is not None:
            path = Path(dataset_path)
            text = path.read_text(encoding="utf-8")
            if path.suffix == ".json":
                payload = json.loads(text)
                if isinstance(payload, dict):
                    rows = payload.get("instances", payload)
                else:
                    rows = payload
                if not isinstance(rows, list):
                    raise ValueError("SWE-bench dataset JSON must contain a list of rows")
                return [self._require_dict(row) for row in rows]
            if path.suffix == ".jsonl":
                return [self._require_dict(json.loads(line)) for line in text.splitlines() if line]
            raise ValueError("SWE-bench dataset path must end with .json or .jsonl")
        try:
            from datasets import load_dataset  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "Loading SWE-bench datasets by name requires the optional 'datasets' package"
            ) from exc
        dataset = load_dataset(dataset_name, split="test")
        return [self._require_dict(row) for row in dataset]

    def _normalize_row(
        self,
        row: dict[str, object],
        *,
        dataset_name: str,
    ) -> SWEbenchLiteInstance:
        instance_id = self._required_str(row, "instance_id")
        repo = self._required_str(row, "repo")
        repo_name = repo.replace("/", "__")
        return SWEbenchLiteInstance(
            instance_id=instance_id,
            repo=repo,
            repo_name=repo_name,
            remote_url=str(row.get("remote_url") or f"https://github.com/{repo}.git"),
            base_commit=self._required_str(row, "base_commit"),
            patch=self._required_str(row, "patch"),
            test_patch=str(row.get("test_patch") or ""),
            problem_statement=self._required_str(row, "problem_statement"),
            hints_text=self._optional_str(row.get("hints_text")),
            fail_to_pass_tests=self._coerce_test_names(row.get("FAIL_TO_PASS")),
            pass_to_pass_tests=self._coerce_test_names(row.get("PASS_TO_PASS")),
            dataset_name=dataset_name,
        )

    def _prepare_repo_cache(
        self,
        instance: SWEbenchLiteInstance,
        repos_root: Path,
    ) -> PreparedRepoCache:
        repo_root = repos_root / instance.repo_name
        if not repo_root.exists():
            self._run(
                ["git", "clone", "--quiet", instance.remote_url, str(repo_root)],
                cwd=None,
            )
        else:
            self._run(
                ["git", "fetch", "--quiet", "--all", "--tags", "--prune"],
                cwd=repo_root,
            )
        default_branch = self._run(
            ["git", "symbolic-ref", "--short", "HEAD"],
            cwd=repo_root,
        ).strip()
        return PreparedRepoCache(repo_root=repo_root, default_branch=default_branch)

    def _prepare_instance_refs(
        self,
        *,
        instance: SWEbenchLiteInstance,
        cache_root: Path,
        repo_cache: PreparedRepoCache,
    ) -> PreparedInstanceRefs:
        base_ref = f"refs/vtm-swebench/{instance.instance_id}/base"
        gold_ref = f"refs/vtm-swebench/{instance.instance_id}/gold"
        self._run(
            ["git", "fetch", "--quiet", "origin", instance.base_commit],
            cwd=repo_cache.repo_root,
        )
        self._run(["git", "update-ref", base_ref, instance.base_commit], cwd=repo_cache.repo_root)
        with tempfile.TemporaryDirectory(dir=cache_root / "worktrees") as temp_dir:
            worktree = Path(temp_dir)
            self._run(
                [
                    "git",
                    "worktree",
                    "add",
                    "--quiet",
                    "--detach",
                    str(worktree),
                    instance.base_commit,
                ],
                cwd=repo_cache.repo_root,
            )
            try:
                self._run(["git", "config", "user.name", "VTM SWE-bench"], cwd=worktree)
                self._run(["git", "config", "user.email", "vtm-swebench@example.com"], cwd=worktree)
                if instance.patch.strip():
                    self._apply_patch(worktree, instance.patch)
                if instance.test_patch.strip():
                    self._apply_patch(worktree, instance.test_patch)
                self._run(["git", "add", "--all"], cwd=worktree)
                self._run(
                    [
                        "git",
                        "commit",
                        "--quiet",
                        "-m",
                        f"Prepare gold ref for {instance.instance_id}",
                    ],
                    cwd=worktree,
                )
                gold_commit = self._run(["git", "rev-parse", "HEAD"], cwd=worktree).strip()
                self._run(["git", "update-ref", gold_ref, gold_commit], cwd=repo_cache.repo_root)
            finally:
                self._run(
                    ["git", "worktree", "remove", "--force", str(worktree)],
                    cwd=repo_cache.repo_root,
                )
        return PreparedInstanceRefs(base_ref=base_ref, gold_ref=gold_ref)

    def _apply_patch(self, repo_root: Path, patch_text: str) -> None:
        normalized_patch = patch_text
        if normalized_patch and not normalized_patch.endswith("\n"):
            normalized_patch += "\n"
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
            patch_path = Path(handle.name)
            handle.write(normalized_patch)
        try:
            self._run(
                ["git", "apply", "--3way", "--whitespace=nowarn", str(patch_path)],
                cwd=repo_root,
            )
        finally:
            patch_path.unlink(missing_ok=True)

    def _coerce_test_names(self, value: object) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return ()
            if stripped.startswith("["):
                parsed = json.loads(stripped)
                return tuple(str(item) for item in parsed)
            return tuple(line.strip() for line in stripped.splitlines() if line.strip())
        if isinstance(value, Iterable):
            return tuple(str(item) for item in value)
        raise ValueError(f"unsupported SWE-bench test list payload: {type(value)!r}")

    def _modified_paths_from_patch(self, patch_text: str) -> tuple[str, ...]:
        paths: list[str] = []
        seen: set[str] = set()
        for line in patch_text.splitlines():
            if not line.startswith("diff --git "):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            candidate = parts[2]
            if candidate.startswith("a/"):
                candidate = candidate[2:]
            if candidate not in seen:
                seen.add(candidate)
                paths.append(candidate)
        return tuple(paths)

    def _sha256(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _run(
        self,
        command: Sequence[str],
        *,
        cwd: Path | None,
    ) -> str:
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout

    def _optional_str(self, value: object) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _required_str(self, row: dict[str, object], key: str) -> str:
        value = row.get(key)
        if value is None:
            raise ValueError(f"SWE-bench row is missing required field: {key}")
        text = str(value).strip()
        if not text:
            raise ValueError(f"SWE-bench row field must be non-empty: {key}")
        return text

    def _require_dict(self, value: object) -> dict[str, object]:
        if not isinstance(value, dict):
            raise ValueError(f"SWE-bench dataset rows must be objects, got {type(value)!r}")
        return value


@dataclass(frozen=True)
class PreparedRepoCache:
    """Cached local clone metadata for a SWE-bench repository."""

    repo_root: Path
    default_branch: str


@dataclass(frozen=True)
class PreparedInstanceRefs:
    """Prepared git refs for a SWE-bench instance."""

    base_ref: str
    gold_ref: str


@dataclass(frozen=True)
class PreparedSWEbenchLiteInstance:
    """Prepared SWE-bench instance plus git refs and digests."""

    instance: SWEbenchLiteInstance
    repo_cache: PreparedRepoCache
    base_ref: str
    gold_ref: str
    expected_changed_paths: tuple[str, ...]
    gold_test_patch_digest: str | None


__all__ = [
    "PreparedSWEbenchLiteInstance",
    "SWEbenchLiteInstance",
    "SWEbenchLitePreparer",
]
