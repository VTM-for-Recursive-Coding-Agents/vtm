from __future__ import annotations

import json
from typing import cast

from vtm.adapters.openai_rlm import OpenAIRLMAdapter
from vtm.adapters.rlm import RLMRankedCandidate, RLMRankRequest
from vtm.enums import ValidityStatus


class FakeResponse:
    def __init__(self, output_text: str) -> None:
        self.output_text = output_text
        self.usage = type(
            "Usage",
            (),
            {"input_tokens": 11, "output_tokens": 7, "total_tokens": 18},
        )()


class FakeResponsesClient:
    def __init__(self) -> None:
        self.last_kwargs: dict[str, object] | None = None

    def create(self, **kwargs: object) -> FakeResponse:
        self.last_kwargs = dict(kwargs)
        return FakeResponse(
            json.dumps(
                {
                    "candidates": [
                        {
                            "candidate_id": "mem_b",
                            "rlm_score": 0.9,
                            "reason": "directly addresses the query",
                        },
                        {
                            "candidate_id": "mem_a",
                            "rlm_score": 0.4,
                            "reason": "still relevant but broader",
                        },
                    ]
                }
            )
        )


class FakeClient:
    def __init__(self) -> None:
        self.responses = FakeResponsesClient()


def test_openai_rlm_adapter_builds_structured_response_request() -> None:
    client = FakeClient()
    adapter = OpenAIRLMAdapter(model="gpt-test", client=client)
    request = RLMRankRequest(
        query="cache invalidation",
        top_k=2,
        candidates=(
            RLMRankedCandidate(
                candidate_id="mem_a",
                title="General cache note",
                summary="Broader cache guidance",
                status=ValidityStatus.VERIFIED,
                lexical_score=0.2,
            ),
            RLMRankedCandidate(
                candidate_id="mem_b",
                title="Specific invalidation note",
                summary="Explains invalidation triggers",
                status=ValidityStatus.RELOCATED,
                lexical_score=0.1,
            ),
        ),
    )

    response = adapter.rank_candidates(request)

    assert client.responses.last_kwargs is not None
    assert client.responses.last_kwargs["model"] == "gpt-test"
    text_config = cast(dict[str, object], client.responses.last_kwargs["text"])
    response_format = cast(dict[str, object], text_config["format"])
    assert response_format["type"] == "json_schema"
    assert response.model_name == "gpt-test"
    assert [candidate.candidate_id for candidate in response.candidates] == ["mem_b", "mem_a"]
    assert response.usage["total_tokens"] == 18
