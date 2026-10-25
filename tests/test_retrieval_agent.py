from types import SimpleNamespace
from unittest.mock import patch

from src.agent import rewrite_queries, retrieve_for_queries


def make_passage(chunk_id, score, text=None):
    return {
        "id": chunk_id,
        "text": text or f"Documentation for {chunk_id}",
        "metadata": {
            "source": f"{chunk_id}.md",
            "heading": "Test section",
        },
        "rerank_score": score,
    }


@patch("src.agent.chat_json")
def test_rewrite_queries_removes_duplicates_and_original_question(mock_chat_json):
    mock_chat_json.return_value.queries = [
        "  How to set a default query parameter?  ",
        "HOW DO I SET A DEFAULT QUERY PARAMETER?",
        "How to define an optional query parameter?",
    ]

    result = rewrite_queries("How do I set a default query parameter?")

    assert result == [
        "How to set a default query parameter?",
        "How to define an optional query parameter?",
    ]
    mock_chat_json.assert_called_once()


@patch("src.agent.search")
def test_retrieve_for_queries_deduplicates_chunks(mock_search):
    first = make_passage("chunk-a", 0.4)
    duplicate = make_passage("chunk-a", 0.8)
    second = make_passage("chunk-b", 0.6)

    mock_search.side_effect = [
        [first, second],
        [duplicate],
    ]

    result = retrieve_for_queries(["query one", "query two"])

    assert len(result) == 2
    assert {item["id"] for item in result} == {"chunk-a", "chunk-b"}
    assert mock_search.call_count == 2


@patch("src.agent.search")
def test_retrieve_for_queries_keeps_highest_score_for_duplicate(mock_search):
    lower = make_passage("chunk-a", 0.3, "Lower-scored version")
    higher = make_passage("chunk-a", 0.9, "Higher-scored version")

    mock_search.side_effect = [[lower], [higher]]

    result = retrieve_for_queries(["query one", "query two"])

    assert len(result) == 1
    assert result[0]["rerank_score"] == 0.9
    assert result[0]["text"] == "Higher-scored version"


@patch("src.agent.TOP_K_FINAL", 2)
@patch("src.agent.search")
def test_retrieve_for_queries_sorts_and_respects_final_limit(mock_search):
    mock_search.return_value = [
        make_passage("chunk-a", 0.2),
        make_passage("chunk-b", 0.9),
        make_passage("chunk-c", 0.5),
    ]

    result = retrieve_for_queries(["query"])

    assert len(result) == 2
    assert [item["id"] for item in result] == ["chunk-b", "chunk-c"]


@patch("src.agent.log_trace")
@patch("src.agent.score_answer")
@patch("src.agent.verify_claims")
@patch("src.agent.grade_context")
@patch("src.agent.retrieve_for_queries")
@patch("src.agent.rewrite_queries")
def test_agent_abstains_when_context_is_insufficient(
    mock_rewrite,
    mock_retrieve,
    mock_grade,
    mock_verify,
    mock_score,
    mock_log,
):
    from src.agent import run_agent

    mock_rewrite.return_value = []
    mock_retrieve.return_value = [
        make_passage("fastapi-auth", 0.9, "Authentication documentation.")
    ]
    mock_grade.return_value = SimpleNamespace(
        sufficient=False,
        reason="The documentation does not contain the requested secret.",
    )
    mock_verify.return_value = SimpleNamespace(claims=[])
    mock_score.return_value = SimpleNamespace(relevance=4, correctness=4)

    with patch("src.agent.generate_answer") as mock_generate:
        result = run_agent(
            "What is the private API key configured on my production server?"
        )

    assert result["context_sufficient"] is False
    assert result["low_confidence"] is True
    assert "can't answer this reliably" in result["answer"].lower()
    assert result["regenerations"] == 0
    mock_generate.assert_not_called()


@patch("src.agent.log_trace")
@patch("src.agent.score_answer")
@patch("src.agent.verify_claims")
@patch("src.agent.grade_context")
@patch("src.agent.retrieve_for_queries")
@patch("src.agent.rewrite_queries")
def test_agent_abstains_on_unknown_server_capacity(
    mock_rewrite,
    mock_retrieve,
    mock_grade,
    mock_verify,
    mock_score,
    mock_log,
):
    from src.agent import run_agent

    mock_rewrite.return_value = []
    mock_retrieve.return_value = [
        make_passage(
            "fastapi-deployment",
            0.9,
            "Deployment documentation.",
        )
    ]
    mock_grade.return_value = SimpleNamespace(
        sufficient=False,
        reason="No exact server capacity is documented.",
    )
    mock_verify.return_value = SimpleNamespace(claims=[])
    mock_score.return_value = SimpleNamespace(relevance=4, correctness=4)

    with patch("src.agent.generate_answer") as mock_generate:
        result = run_agent(
            "What exact number of concurrent requests can my production server handle?"
        )

    assert result["context_sufficient"] is False
    assert result["low_confidence"] is True
    assert "can't answer this reliably" in result["answer"].lower()
    assert result["regenerations"] == 0
    mock_generate.assert_not_called()