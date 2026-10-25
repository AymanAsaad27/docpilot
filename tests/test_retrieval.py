from unittest.mock import Mock

import pytest

from src import retrieval


@pytest.fixture
def sample_candidates():
    return [
        {
            "id": "doc-1",
            "text": "FastAPI supports dependency injection.",
            "metadata": {"source": "dependencies.md"},
            "vector_distance": 0.2,
            "vector_score": 0.8,
        },
        {
            "id": "doc-2",
            "text": "FastAPI supports request validation.",
            "metadata": {"source": "requests.md"},
            "vector_distance": 0.4,
            "vector_score": 0.6,
        },
    ]


def test_get_collection_opens_persistent_collection(monkeypatch):
    retrieval.get_collection.cache_clear()

    collection = Mock()
    client = Mock()
    client.get_collection.return_value = collection

    monkeypatch.setattr(
        retrieval.chromadb,
        "PersistentClient",
        lambda path: client,
    )

    try:
        result = retrieval.get_collection()

        assert result is collection
        client.get_collection.assert_called_once_with(retrieval.COLLECTION)
    finally:
        retrieval.get_collection.cache_clear()


def test_get_reranker_loads_configured_model(monkeypatch):
    retrieval.get_reranker.cache_clear()

    model = Mock()
    constructor = Mock(return_value=model)
    monkeypatch.setattr(retrieval, "CrossEncoder", constructor)

    try:
        assert retrieval.get_reranker() is model
        constructor.assert_called_once_with(retrieval.RERANK_MODEL)

        # The cached call should not load the model again.
        assert retrieval.get_reranker() is model
        constructor.assert_called_once()
    finally:
        retrieval.get_reranker.cache_clear()


def test_retrieve_candidates_returns_empty_for_blank_query(monkeypatch):
    collection_getter = Mock()
    embedding = Mock()

    monkeypatch.setattr(retrieval, "get_collection", collection_getter)
    monkeypatch.setattr(retrieval, "embed", embedding)

    assert retrieval.retrieve_candidates("   \n ") == []

    collection_getter.assert_not_called()
    embedding.assert_not_called()


def test_retrieve_candidates_builds_results_and_scores(monkeypatch):
    collection = Mock()
    collection.count.return_value = 3
    collection.query.return_value = {
        "ids": [["doc-1", "doc-2"]],
        "documents": [["First passage", "Second passage"]],
        "metadatas": [
            [{"source": "one.md"}, {"source": "two.md"}]
        ],
        "distances": [[0.25, 0.7]],
    }

    monkeypatch.setattr(retrieval, "get_collection", lambda: collection)
    embed_mock = Mock(return_value=[[0.1, 0.2, 0.3]])
    monkeypatch.setattr(retrieval, "embed", embed_mock)

    results = retrieval.retrieve_candidates("  FastAPI question  ", top_k=10)

    embed_mock.assert_called_once_with(["FastAPI question"])
    collection.query.assert_called_once_with(
        query_embeddings=[[0.1, 0.2, 0.3]],
        n_results=3,
        include=["documents", "metadatas", "distances"],
    )

    assert len(results) == 2
    assert results[0] == {
        "id": "doc-1",
        "text": "First passage",
        "metadata": {"source": "one.md"},
        "vector_distance": 0.25,
        "vector_score": 0.75,
    }
    assert results[1]["vector_score"] == pytest.approx(0.3)


def test_retrieve_candidates_limits_results_to_top_k(monkeypatch):
    collection = Mock()
    collection.count.return_value = 50
    collection.query.return_value = {
        "ids": [[]],
        "documents": [[]],
        "metadatas": [[]],
        "distances": [[]],
    }

    monkeypatch.setattr(retrieval, "get_collection", lambda: collection)
    monkeypatch.setattr(retrieval, "embed", lambda texts: [[0.1]])

    retrieval.retrieve_candidates("question", top_k=7)

    assert collection.query.call_args.kwargs["n_results"] == 7


def test_rerank_candidates_returns_empty_for_no_candidates(monkeypatch):
    reranker_getter = Mock()
    monkeypatch.setattr(retrieval, "get_reranker", reranker_getter)

    assert retrieval.rerank_candidates("question", []) == []

    reranker_getter.assert_not_called()


def test_rerank_candidates_sorts_scores_and_preserves_originals(
    monkeypatch, sample_candidates
):
    reranker = Mock()
    reranker.predict.return_value = [0.2, 0.9]
    monkeypatch.setattr(retrieval, "get_reranker", lambda: reranker)

    results = retrieval.rerank_candidates(
        "FastAPI question",
        sample_candidates,
        top_k=2,
    )

    reranker.predict.assert_called_once_with(
        [
            ("FastAPI question", sample_candidates[0]["text"]),
            ("FastAPI question", sample_candidates[1]["text"]),
        ],
        show_progress_bar=False,
    )

    assert [item["id"] for item in results] == ["doc-2", "doc-1"]
    assert results[0]["rerank_score"] == pytest.approx(0.9)
    assert results[1]["rerank_score"] == pytest.approx(0.2)

    # Reranking copies candidates instead of modifying the originals.
    assert "rerank_score" not in sample_candidates[0]
    assert "rerank_score" not in sample_candidates[1]


def test_rerank_candidates_respects_final_limit(
    monkeypatch, sample_candidates
):
    reranker = Mock()
    reranker.predict.return_value = [0.2, 0.9]
    monkeypatch.setattr(retrieval, "get_reranker", lambda: reranker)

    results = retrieval.rerank_candidates(
        "question",
        sample_candidates,
        top_k=1,
    )

    assert len(results) == 1
    assert results[0]["id"] == "doc-2"


def test_search_passes_parameters_through_both_stages(
    monkeypatch, sample_candidates
):
    candidates = Mock(return_value=sample_candidates)
    reranked = Mock(return_value=[sample_candidates[1]])

    monkeypatch.setattr(retrieval, "retrieve_candidates", candidates)
    monkeypatch.setattr(retrieval, "rerank_candidates", reranked)

    result = retrieval.search(
        "my question",
        retrieve_k=12,
        final_k=3,
    )

    candidates.assert_called_once_with("my question", top_k=12)
    reranked.assert_called_once_with(
        "my question",
        sample_candidates,
        top_k=3,
    )
    assert result == [sample_candidates[1]]


def test_search_returns_empty_when_retrieval_finds_nothing(monkeypatch):
    monkeypatch.setattr(
        retrieval,
        "retrieve_candidates",
        Mock(return_value=[]),
    )
    rerank = Mock(return_value=[])
    monkeypatch.setattr(retrieval, "rerank_candidates", rerank)

    assert retrieval.search("question") == []

    rerank.assert_called_once_with(
        "question",
        [],
        top_k=retrieval.TOP_K_FINAL,
    )
