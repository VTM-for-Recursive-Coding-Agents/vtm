from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import timedelta
from typing import cast

from vtm.adapters.git import GitFingerprintAdapter, GitRepoFingerprintCollector
from vtm.adapters.rlm import RLMAdapter, RLMRankedCandidate, RLMRankRequest, RLMRankResponse
from vtm.adapters.runtime import EnvFingerprintAdapter, RuntimeEnvFingerprintCollector
from vtm.base import utc_now
from vtm.cache import CacheEntry, CacheKey
from vtm.evidence import EvidenceRef
from vtm.memory_items import MemoryItem
from vtm.retrieval import RetrieveCandidate, RetrieveExplanation, RetrieveRequest, RetrieveResult
from vtm.services.retriever import Retriever
from vtm.stores.base import CacheStore


class RLMRerankingRetriever:
    def __init__(
        self,
        base_retriever: Retriever,
        rlm_adapter: RLMAdapter,
        *,
        top_k_lexical: int = 20,
        top_k_final: int = 10,
        cache_store: CacheStore | None = None,
        cache_repo_root: str | None = None,
        repo_collector: GitFingerprintAdapter | None = None,
        env_collector: EnvFingerprintAdapter | None = None,
        tool_probes: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        if top_k_lexical <= 0:
            raise ValueError("top_k_lexical must be greater than zero")
        if top_k_final <= 0:
            raise ValueError("top_k_final must be greater than zero")
        if top_k_final > top_k_lexical:
            raise ValueError("top_k_final must be less than or equal to top_k_lexical")

        self._base_retriever = base_retriever
        self._rlm_adapter = rlm_adapter
        self._top_k_lexical = top_k_lexical
        self._top_k_final = top_k_final
        self._cache_store = cache_store
        self._cache_repo_root = cache_repo_root
        self._repo_collector = repo_collector or GitRepoFingerprintCollector()
        self._env_collector = env_collector or RuntimeEnvFingerprintCollector()
        self._tool_probes = {
            name: tuple(command) for name, command in (tool_probes or {}).items()
        }

    def retrieve(self, request: RetrieveRequest) -> RetrieveResult:
        lexical_limit = max(request.limit, self._top_k_lexical)
        lexical_request = request.model_copy(update={"limit": lexical_limit})
        lexical_result = self._base_retriever.retrieve(lexical_request)
        lexical_candidates = lexical_result.candidates[: self._top_k_lexical]
        final_limit = min(request.limit, self._top_k_final)

        if not lexical_candidates:
            return lexical_result.model_copy(update={"candidates": ()})

        rank_request = self._build_rank_request(request, lexical_candidates, final_limit)
        rank_response: RLMRankResponse | None = self._load_cached_response(rank_request)
        cache_hit = rank_response is not None
        if rank_response is None:
            try:
                rank_response = self._rlm_adapter.rank_candidates(rank_request)
            except Exception as exc:
                return self._limit_lexical_result(
                    lexical_result,
                    final_limit,
                    error=str(exc),
                )
            self._save_cached_response(rank_request, rank_response)

        reranked = self._merge_candidates(
            lexical_candidates,
            rank_response,
            final_limit=final_limit,
            cache_hit=cache_hit,
        )
        return lexical_result.model_copy(
            update={
                "request": request,
                "candidates": tuple(reranked),
            }
        )

    def expand(self, memory_id: str) -> tuple[EvidenceRef, ...]:
        return self._base_retriever.expand(memory_id)

    def _build_rank_request(
        self,
        request: RetrieveRequest,
        candidates: Sequence[RetrieveCandidate],
        final_limit: int,
    ) -> RLMRankRequest:
        ranked_candidates = []
        for candidate in candidates:
            anchor_path, anchor_symbol = self._anchor_metadata(candidate.memory)
            ranked_candidates.append(
                RLMRankedCandidate(
                    candidate_id=candidate.memory.memory_id,
                    title=candidate.memory.title,
                    summary=candidate.memory.summary,
                    tags=candidate.memory.tags,
                    status=candidate.memory.validity.status,
                    path=anchor_path,
                    symbol=anchor_symbol,
                    lexical_score=candidate.score,
                )
            )

        return RLMRankRequest(
            query=request.query,
            candidates=tuple(ranked_candidates),
            top_k=final_limit,
            metadata={
                "detail_level": request.detail_level.value,
                "evidence_budget": request.evidence_budget.value,
                "scope_ids": [scope.scope_id for scope in request.scopes],
            },
        )

    def _load_cached_response(self, request: RLMRankRequest) -> RLMRankResponse | None:
        cache_key = self._build_cache_key(request)
        if cache_key is None or self._cache_store is None:
            return None
        entry = self._cache_store.get_cache_entry(cache_key)
        if entry is None:
            return None
        try:
            return RLMRankResponse.model_validate(entry.value)
        except Exception:
            return None

    def _save_cached_response(self, request: RLMRankRequest, response: RLMRankResponse) -> None:
        cache_key = self._build_cache_key(request)
        if cache_key is None or self._cache_store is None:
            return
        self._cache_store.save_cache_entry(
            CacheEntry(
                key=cache_key,
                value=response.model_dump(mode="json"),
                expires_at=utc_now() + timedelta(days=7),
            )
        )

    def _build_cache_key(self, request: RLMRankRequest) -> CacheKey | None:
        if self._cache_store is None or self._cache_repo_root is None:
            return None
        repo_fingerprint = self._repo_collector.collect(self._cache_repo_root)
        env_fingerprint = self._env_collector.collect(tool_probes=self._tool_probes)
        candidate_digest = hashlib.sha256(
            json.dumps(
                [candidate.model_dump(mode="json") for candidate in request.candidates],
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        cache_identity = cast(
            str,
            getattr(self._rlm_adapter, "cache_identity", type(self._rlm_adapter).__name__),
        )
        return CacheKey.from_parts(
            "rlm_rerank",
            {
                "query": request.query,
                "candidate_digest": candidate_digest,
                "top_k": request.top_k,
                "adapter": cache_identity,
            },
            repo_fingerprint,
            env_fingerprint,
        )

    def _merge_candidates(
        self,
        lexical_candidates: Sequence[RetrieveCandidate],
        response: RLMRankResponse,
        *,
        final_limit: int,
        cache_hit: bool,
    ) -> list[RetrieveCandidate]:
        lexical_map = {
            candidate.memory.memory_id: candidate for candidate in lexical_candidates
        }
        ranked_candidates: list[RetrieveCandidate] = []
        seen: set[str] = set()
        for ranked in response.candidates:
            lexical_candidate = lexical_map.get(ranked.candidate_id)
            if lexical_candidate is None or ranked.candidate_id in seen:
                continue
            seen.add(ranked.candidate_id)
            ranked_candidates.append(
                self._update_candidate(
                    lexical_candidate,
                    rlm_score=ranked.rlm_score,
                    final_score=ranked.final_score or ranked.rlm_score or lexical_candidate.score,
                    reason=ranked.reason,
                    cache_hit=cache_hit,
                    reranked=True,
                )
            )

        for lexical_candidate in lexical_candidates:
            if lexical_candidate.memory.memory_id in seen:
                continue
            ranked_candidates.append(
                self._update_candidate(
                    lexical_candidate,
                    rlm_score=lexical_candidate.score,
                    final_score=lexical_candidate.score,
                    reason="preserved lexical order after rerank response",
                    cache_hit=cache_hit,
                    reranked=False,
                )
            )

        return ranked_candidates[:final_limit]

    def _limit_lexical_result(
        self,
        lexical_result: RetrieveResult,
        final_limit: int,
        *,
        error: str,
    ) -> RetrieveResult:
        fallback_candidates = []
        for candidate in lexical_result.candidates[:final_limit]:
            metadata = {
                **candidate.explanation.metadata,
                "lexical_score": candidate.score,
                "rlm_error": error,
                "reranked": False,
            }
            fallback_candidates.append(
                candidate.model_copy(
                    update={
                        "explanation": candidate.explanation.model_copy(
                            update={"metadata": metadata}
                        )
                    }
                )
            )
        return lexical_result.model_copy(update={"candidates": tuple(fallback_candidates)})

    def _update_candidate(
        self,
        candidate: RetrieveCandidate,
        *,
        rlm_score: float | None,
        final_score: float,
        reason: str | None,
        cache_hit: bool,
        reranked: bool,
    ) -> RetrieveCandidate:
        metadata = {
            **candidate.explanation.metadata,
            "lexical_score": candidate.score,
            "rlm_score": rlm_score,
            "final_score": final_score,
            "reranked": reranked,
            "rlm_cache_hit": cache_hit,
        }
        updated_explanation = RetrieveExplanation(
            matched_tokens=candidate.explanation.matched_tokens,
            matched_fields=candidate.explanation.matched_fields,
            score=final_score,
            reason=reason or candidate.explanation.reason,
            metadata=metadata,
        )
        return candidate.model_copy(
            update={"score": final_score, "explanation": updated_explanation}
        )

    def _anchor_metadata(self, item: MemoryItem) -> tuple[str | None, str | None]:
        for evidence in item.evidence:
            if evidence.code_anchor is not None:
                return evidence.code_anchor.path, evidence.code_anchor.symbol
        return None, None
