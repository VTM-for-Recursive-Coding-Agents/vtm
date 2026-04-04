from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path

from vtm.adapters.rlm import RLMRankRequest, RLMRankResponse
from vtm.enums import ValidityStatus
from vtm.memory_items import MemoryItem, ValidityState, VisibilityScope
from vtm.retrieval import RetrieveRequest
from vtm.services.reranking_retriever import RLMRerankingRetriever
from vtm.services.retriever import LexicalRetriever
from vtm.stores.cache_store import SqliteCacheStore
from vtm.stores.sqlite_store import SqliteMetadataStore


class FakeRLMAdapter:
    def __init__(self) -> None:
        self.calls = 0
        self.cache_identity = "fake-rlm:test"

    def rank_candidates(self, request: RLMRankRequest) -> RLMRankResponse:
        self.calls += 1
        ranked = tuple(
            candidate.model_copy(
                update={
                    "rlm_score": float(index),
                    "final_score": float(index),
                    "reason": f"reranked-{candidate.candidate_id}",
                }
            )
            for index, candidate in enumerate(reversed(request.candidates), start=1)
        )
        return RLMRankResponse(candidates=ranked, model_name="fake-model")


class FailingRLMAdapter:
    def rank_candidates(self, request: RLMRankRequest) -> RLMRankResponse:
        raise RuntimeError("simulated rerank failure")


def _init_repo(repo: Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.name", "VTM Tests"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "vtm@example.com"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    (repo / "tracked.txt").write_text("tracked\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=repo, check=True, capture_output=True)


def test_rlm_reranking_retriever_preserves_expand_and_records_scores(
    metadata_store: SqliteMetadataStore,
    memory_factory: Callable[..., MemoryItem],
    scope: VisibilityScope,
) -> None:
    first = memory_factory(title="Parser alpha", summary="parser alpha summary")
    second = memory_factory(title="Parser beta", summary="parser beta summary")
    metadata_store.save_memory_item(first)
    metadata_store.save_memory_item(second)

    adapter = FakeRLMAdapter()
    lexical = LexicalRetriever(metadata_store)
    lexical_result = lexical.retrieve(RetrieveRequest(query="parser", scopes=(scope,), limit=2))
    lexical_ids = [candidate.memory.memory_id for candidate in lexical_result.candidates]
    lexical_scores = {
        candidate.memory.memory_id: candidate.score for candidate in lexical_result.candidates
    }
    retriever = RLMRerankingRetriever(
        lexical,
        adapter,
        top_k_lexical=5,
        top_k_final=2,
    )

    result = retriever.retrieve(RetrieveRequest(query="parser", scopes=(scope,), limit=2))

    assert adapter.calls == 1
    assert len(result.candidates) == 2
    assert [candidate.memory.memory_id for candidate in result.candidates] == list(
        reversed(lexical_ids)
    )
    top_memory_id = result.candidates[0].memory.memory_id
    assert (
        result.candidates[0].explanation.metadata["lexical_score"]
        == lexical_scores[top_memory_id]
    )
    assert result.candidates[0].explanation.metadata["reranked"] is True
    assert result.candidates[0].score == 1.0
    assert retriever.expand(first.memory_id) == first.evidence


def test_rlm_reranking_retriever_preserves_lexical_order_on_failure(
    metadata_store: SqliteMetadataStore,
    memory_factory: Callable[..., MemoryItem],
    scope: VisibilityScope,
) -> None:
    first = memory_factory(title="Parser alpha", summary="parser alpha summary")
    second = memory_factory(title="Parser beta", summary="parser beta summary")
    metadata_store.save_memory_item(first)
    metadata_store.save_memory_item(second)

    lexical = LexicalRetriever(metadata_store)
    lexical_result = lexical.retrieve(RetrieveRequest(query="parser", scopes=(scope,), limit=2))
    lexical_ids = [candidate.memory.memory_id for candidate in lexical_result.candidates]
    retriever = RLMRerankingRetriever(lexical, FailingRLMAdapter(), top_k_lexical=5, top_k_final=2)

    result = retriever.retrieve(RetrieveRequest(query="parser", scopes=(scope,), limit=2))

    assert [candidate.memory.memory_id for candidate in result.candidates] == lexical_ids
    assert result.candidates[0].explanation.metadata["reranked"] is False
    assert "simulated rerank failure" in result.candidates[0].explanation.metadata["rlm_error"]


def test_reranking_cache_hits_and_repo_drift_invalidates(
    tmp_path: Path,
    metadata_store: SqliteMetadataStore,
    cache_store: SqliteCacheStore,
    memory_factory: Callable[..., MemoryItem],
    scope: VisibilityScope,
) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)

    memory = memory_factory(
        title="Cached parser claim",
        summary="cached parser summary",
        validity_status=ValidityStatus.VERIFIED,
    )
    memory = memory.model_copy(
        update={
            "validity": ValidityState(
                status=ValidityStatus.VERIFIED,
                dependency_fingerprint=memory.validity.dependency_fingerprint,
            )
        }
    )
    metadata_store.save_memory_item(memory)

    adapter = FakeRLMAdapter()
    retriever = RLMRerankingRetriever(
        LexicalRetriever(metadata_store),
        adapter,
        top_k_lexical=5,
        top_k_final=1,
        cache_store=cache_store,
        cache_repo_root=str(repo),
    )

    retriever.retrieve(RetrieveRequest(query="cached", scopes=(scope,), limit=1))
    second = retriever.retrieve(RetrieveRequest(query="cached", scopes=(scope,), limit=1))

    assert adapter.calls == 1
    assert second.candidates[0].explanation.metadata["rlm_cache_hit"] is True

    (repo / "tracked.txt").write_text("changed\n", encoding="utf-8")
    third = retriever.retrieve(RetrieveRequest(query="cached", scopes=(scope,), limit=1))

    assert adapter.calls == 2
    assert third.candidates[0].explanation.metadata["rlm_cache_hit"] is False
