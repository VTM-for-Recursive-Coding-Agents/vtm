from __future__ import annotations

import json
import os
from typing import Any

from vtm.adapters.rlm import RLMAdapter, RLMRankedCandidate, RLMRankRequest, RLMRankResponse


class OpenAIRLMAdapter:
    def __init__(
        self,
        *,
        model: str,
        client: Any | None = None,
        api_key: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        max_output_tokens: int = 1200,
        temperature: float = 0.0,
    ) -> None:
        if not model:
            raise ValueError("OpenAI RLM adapter requires a non-empty model name")
        if max_output_tokens <= 0:
            raise ValueError("OpenAI RLM adapter requires max_output_tokens > 0")

        self._model = model
        self._max_output_tokens = max_output_tokens
        self._temperature = temperature
        self._client = client or self._build_client(api_key=api_key, api_key_env=api_key_env)

    @property
    def cache_identity(self) -> str:
        return f"openai:{self._model}"

    def rank_candidates(self, request: RLMRankRequest) -> RLMRankResponse:
        if not request.candidates:
            return RLMRankResponse(model_name=self._model)

        response = self._client.responses.create(
            model=self._model,
            input=self._build_prompt(request),
            max_output_tokens=self._max_output_tokens,
            temperature=self._temperature,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "vtm_rerank_response",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "candidates": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "properties": {
                                        "candidate_id": {"type": "string"},
                                        "rlm_score": {"type": "number"},
                                        "reason": {"type": "string"},
                                    },
                                    "required": ["candidate_id", "rlm_score", "reason"],
                                },
                            }
                        },
                        "required": ["candidates"],
                    },
                    "description": "Rerank lexical retrieval candidates for repository memory.",
                },
                "verbosity": "low",
            },
        )
        payload = json.loads(self._extract_output_text(response))
        ranked_payload = payload.get("candidates")
        if not isinstance(ranked_payload, list):
            raise ValueError("OpenAI RLM adapter returned an invalid candidate payload")

        lexical_map = {candidate.candidate_id: candidate for candidate in request.candidates}
        ranked_candidates: list[RLMRankedCandidate] = []
        seen: set[str] = set()
        for raw_candidate in ranked_payload:
            if not isinstance(raw_candidate, dict):
                raise ValueError("OpenAI RLM adapter returned a non-object candidate")
            candidate_id = raw_candidate.get("candidate_id")
            rlm_score = raw_candidate.get("rlm_score")
            reason = raw_candidate.get("reason")
            if (
                not isinstance(candidate_id, str)
                or candidate_id not in lexical_map
                or not isinstance(rlm_score, (int, float))
                or not isinstance(reason, str)
            ):
                raise ValueError("OpenAI RLM adapter returned an unknown or malformed candidate")
            if candidate_id in seen:
                raise ValueError("OpenAI RLM adapter returned duplicate candidate ids")
            seen.add(candidate_id)
            lexical_candidate = lexical_map[candidate_id]
            ranked_candidates.append(
                lexical_candidate.model_copy(
                    update={
                        "rlm_score": float(rlm_score),
                        "final_score": float(rlm_score),
                        "reason": reason,
                    }
                )
            )

        for lexical_candidate in request.candidates:
            if lexical_candidate.candidate_id in seen:
                continue
            ranked_candidates.append(
                lexical_candidate.model_copy(
                    update={
                        "rlm_score": lexical_candidate.lexical_score,
                        "final_score": lexical_candidate.lexical_score,
                        "reason": "not explicitly reranked by model; preserving lexical order",
                    }
                )
            )

        return RLMRankResponse(
            candidates=tuple(ranked_candidates[: request.top_k]),
            model_name=self._model,
            usage=self._extract_usage(response),
            metadata={"provider": "openai"},
        )

    def _build_client(self, *, api_key: str | None, api_key_env: str) -> Any:
        resolved_api_key = api_key or os.getenv(api_key_env)
        if not resolved_api_key:
            raise ValueError(
                "OpenAI RLM adapter requires api_key or the environment variable "
                f"{api_key_env!r}"
            )
        try:
            from openai import OpenAI  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "OpenAI RLM adapter requires the optional 'openai' dependency"
            ) from exc
        return OpenAI(api_key=resolved_api_key)

    def _build_prompt(self, request: RLMRankRequest) -> str:
        candidate_lines = []
        for index, candidate in enumerate(request.candidates, start=1):
            candidate_lines.append(
                {
                    "rank_hint": index,
                    "candidate_id": candidate.candidate_id,
                    "title": candidate.title,
                    "summary": candidate.summary,
                    "tags": list(candidate.tags),
                    "status": candidate.status.value,
                    "path": candidate.path,
                    "symbol": candidate.symbol,
                    "lexical_score": candidate.lexical_score,
                }
            )

        return "\n".join(
            [
                "You rerank repository memory candidates for coding workflows.",
                "Return candidates ordered best-to-worst for the query.",
                "Use only the provided candidate fields.",
                "Prefer candidates that are directly relevant, verified, and specific.",
                "",
                f"Query: {request.query}",
                f"Top K: {request.top_k}",
                "Candidates JSON:",
                json.dumps(candidate_lines, indent=2, sort_keys=True),
            ]
        )

    def _extract_output_text(self, response: Any) -> str:
        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str) and output_text.strip():
            return output_text
        raise ValueError("OpenAI RLM adapter did not return structured output text")

    def _extract_usage(self, response: Any) -> dict[str, Any]:
        usage = getattr(response, "usage", None)
        if usage is None:
            return {}
        usage_fields = ("input_tokens", "output_tokens", "total_tokens")
        extracted = {
            field_name: getattr(usage, field_name)
            for field_name in usage_fields
            if hasattr(usage, field_name)
        }
        return {key: value for key, value in extracted.items() if value is not None}


__all__ = ["OpenAIRLMAdapter", "RLMAdapter", "RLMRankRequest", "RLMRankResponse"]
