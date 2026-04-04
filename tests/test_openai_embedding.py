from __future__ import annotations

from vtm.adapters.openai_embedding import OpenAIEmbeddingAdapter


class FakeEmbeddingResponse:
    def __init__(self) -> None:
        self.data = [type("EmbeddingRecord", (), {"embedding": [0.25, -0.5, 0.75]})()]


class FakeEmbeddingsClient:
    def __init__(self) -> None:
        self.last_kwargs: dict[str, object] | None = None

    def create(self, **kwargs: object) -> FakeEmbeddingResponse:
        self.last_kwargs = dict(kwargs)
        return FakeEmbeddingResponse()


class FakeClient:
    def __init__(self) -> None:
        self.embeddings = FakeEmbeddingsClient()


def test_openai_embedding_adapter_builds_embedding_request() -> None:
    client = FakeClient()
    adapter = OpenAIEmbeddingAdapter(model="text-embedding-test", client=client)

    vector = adapter.embed_text("cache invalidation note")

    assert client.embeddings.last_kwargs is not None
    assert client.embeddings.last_kwargs["model"] == "text-embedding-test"
    assert client.embeddings.last_kwargs["input"] == "cache invalidation note"
    assert vector == (0.25, -0.5, 0.75)
    assert adapter.adapter_id == "openai:text-embedding-test"
